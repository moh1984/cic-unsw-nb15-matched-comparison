#!/usr/bin/env python3
"""
Combine independent disjoint-block runs into confidence intervals that do not
rely on the fold scores being independent.

Repeated cross-validation on a single sample reuses the same records across
folds and repetitions, so the fifteen fold scores are dependent and an interval
computed from them is too narrow. Running the pipeline once per disjoint block
(`ids_evaluation.py --disjoint-blocks N --block K`) yields one estimate per
block from non-overlapping records. The estimates are then independent, and a
Student-t interval across blocks is valid.

Usage:
    # first produce the runs, one per block
    for K in 0 1 2 3 4; do
        python src/ids_evaluation.py --cic ... --unsw ... \
            --config config/config.json --outdir results/block$K \
            --disjoint-blocks 5 --block $K --sample 100000 \
            --models RF,XGB,DT --n-splits 5 --n-repeats 1 --skip-shap
    done

    # then combine
    python src/aggregate_independent_runs.py \
        --runs results/block0 results/block1 results/block2 results/block3 results/block4 \
        --model XGB --out results/independent_ci.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = [("", "Baseline"), ("E1_", "E1 matched"), ("E2_", "E2 common")]
METRICS = ["accuracy", "balanced_accuracy", "recall_attack", "macro_f1", "MCC", "FAR"]


def run_mean(run: Path, tag: str, model: str, metric: str) -> float | None:
    """The point estimate of one independent run: the mean over its own folds."""
    path = run / f"folds_{tag}_{model}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return float(df[metric].mean()) if metric in df.columns else None


def t_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, df=n - 1) * sem
    return mean, mean - half, mean + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one output directory per disjoint block")
    ap.add_argument("--model", default="XGB")
    ap.add_argument("--dataset-a", default="CIC")
    ap.add_argument("--dataset-b", default="UNSW")
    ap.add_argument("--out", default="independent_ci.csv")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    if len(runs) < 3:
        print("WARNING: fewer than three blocks gives a very wide t interval.")

    rows = []
    for prefix, label in CONDITIONS:
        for metric in METRICS:
            a = [run_mean(r, f"{prefix}{args.dataset_a}", args.model, metric) for r in runs]
            b = [run_mean(r, f"{prefix}{args.dataset_b}", args.model, metric) for r in runs]
            paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(paired) < 2:
                continue
            arr_a = np.array([x for x, _ in paired])
            arr_b = np.array([y for _, y in paired])
            diffs = arr_a - arr_b

            mean, lo, hi = t_interval(diffs)
            # paired t-test across blocks: valid, since blocks are disjoint
            t_stat, p = stats.ttest_rel(arr_a, arr_b) if len(paired) > 1 else (np.nan, np.nan)

            rows.append({
                "condition": label,
                "metric": metric,
                "n_blocks": len(paired),
                f"mean_{args.dataset_a}": arr_a.mean(),
                f"mean_{args.dataset_b}": arr_b.mean(),
                "difference": mean,
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "p_paired_t": float(p),
                "per_block_differences": np.round(diffs, 5).tolist(),
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df[["condition", "metric", "n_blocks", "difference", "ci_low", "ci_high",
              "ci_excludes_zero", "p_paired_t"]].to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nwritten -> {args.out}")
    print("\nThese intervals are computed across disjoint blocks, so they do not rely "
          "on the fold-score independence assumption discussed in Section 3.4 of the "
          "paper. Expect them to be WIDER than the bootstrap intervals of Table 9; "
          "that is the point.")


if __name__ == "__main__":
    main()
