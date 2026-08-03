#!/usr/bin/env python3
"""
Between-dataset significance analysis (Table 12 of the paper).

McNemar's test cannot be used to compare the two datasets, because they contain
different records and the predictions are therefore not paired. This script
instead compares the two sets of fifteen cross-validation fold scores using:

  * Mann-Whitney U      -- no distributional assumption
  * Cliff's delta       -- effect size; +/-1.0 means the two sets of fold
                           scores do not overlap at any point
  * percentile bootstrap -- 95% confidence interval on the difference of means

Caveat, reported in the paper and repeated here: the fifteen fold scores arise
from three repetitions over the same records and are therefore not mutually
independent. Any test treating them as an independent sample understates the
true uncertainty, so the p-values are descriptive rather than exact. The
assumption-free result is the overlap column.

Usage:
    python significance_tests.py --results results/main --model XGB \
        --out results/main/dataset_significance.csv
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RANDOM_STATE = 42
CONDITIONS = [("", "Baseline"), ("E1_", "E1 matched"), ("E2_", "E2 common")]
METRICS = ["accuracy", "balanced_accuracy", "recall_attack", "macro_f1", "MCC", "FAR"]


def fold_scores(results: Path, tag: str, model: str, metric: str) -> np.ndarray:
    path = results / f"folds_{tag}_{model}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise SystemExit(f"{path} has no column '{metric}'")
    return df[metric].to_numpy(dtype=float)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b). +/-1.0 means the samples do not overlap."""
    greater = sum(x > y for x, y in product(a, b))
    less = sum(x < y for x, y in product(a, b))
    return (greater - less) / (len(a) * len(b))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 20_000,
                 alpha: float = 0.05, seed: int = RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(a, (n_boot, len(a)), replace=True).mean(axis=1)
             - rng.choice(b, (n_boot, len(b)), replace=True).mean(axis=1))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--model", default="XGB")
    ap.add_argument("--dataset-a", default="CIC", help="tag prefix of the first dataset")
    ap.add_argument("--dataset-b", default="UNSW")
    ap.add_argument("--n-boot", type=int, default=20_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = Path(args.results)
    rows = []
    for prefix, label in CONDITIONS:
        for metric in METRICS:
            try:
                a = fold_scores(results, f"{prefix}{args.dataset_a}", args.model, metric)
                b = fold_scores(results, f"{prefix}{args.dataset_b}", args.model, metric)
            except SystemExit as exc:
                print(f"  skipping {label}/{metric}: {exc}")
                continue

            _, p = mannwhitneyu(a, b, alternative="two-sided")
            lo, hi = bootstrap_ci(a, b, args.n_boot)
            overlap = not (a.min() > b.max() or b.min() > a.max())

            rows.append({
                "condition": label,
                "metric": metric,
                f"mean_{args.dataset_a}": a.mean(),
                f"mean_{args.dataset_b}": b.mean(),
                "difference": a.mean() - b.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "cliffs_delta": cliffs_delta(a, b),
                "distributions_overlap": overlap,
                "p_mann_whitney": p,
                f"range_{args.dataset_a}": f"[{a.min():.4f}, {a.max():.4f}]",
                f"range_{args.dataset_b}": f"[{b.min():.4f}, {b.max():.4f}]",
                "n_folds": len(a),
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df[["condition", "metric", "difference", "ci_low", "ci_high",
              "cliffs_delta", "distributions_overlap", "p_mann_whitney"]]
          .to_string(index=False))

    out = Path(args.out) if args.out else results / "dataset_significance.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten -> {out}")

    n_sep = int((df["cliffs_delta"].abs() == 1.0).sum())
    print(f"{n_sep} of {len(df)} comparisons show complete non-overlap "
          f"(|Cliff's delta| = 1.0)")
    print("NOTE: fold scores are not mutually independent; p-values are descriptive.")


if __name__ == "__main__":
    main()
