#!/usr/bin/env python3
"""
E2 sensitivity analysis: does the common-feature result depend on the
alignment choices of Table 4?

Two variants are run and compared against the baseline E2 result:

  no_window  drops the two TCP-window pairs (swin/FWD Init Win Bytes and
             dwin/Bwd Init Win Bytes), the weakest correspondence in Table 4,
             leaving ten pairs.
  swapped    reverses the forward/backward convention on the CICFlowMeter side.
             WARNING: this is a null test by construction and its output must
             not be reported as evidence. Each classifier is trained on the
             columns of its own dataset, so exchanging which column of one
             dataset is deemed to correspond to which column of the other
             leaves both feature sets identical. Any observed difference comes
             from column ordering affecting tie-breaking, not from the
             convention. It is retained only to document this fact: the
             direction convention cannot affect E2 at all.

Only XGBoost is run, since the question is about the feature alignment rather
than about model choice.

Usage:
    python run_e2_sensitivity.py \
        --cic data/CIC_NB15.csv --unsw data/unsw_all.csv \
        --baseline results/main --outdir results/e2_sensitivity \
        --sample 100000

Runtime: roughly 25 minutes on two CPU cores for both variants.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

VARIANTS = [
    ("no_window", "config_E2_no_window.json", "10 pairs; TCP-window pairs dropped"),
    ("swapped", "config_E2_swapped.json", "12 pairs; fwd/bwd convention reversed"),
]
METRICS = ["accuracy", "balanced_accuracy", "recall_attack", "macro_f1", "MCC", "FAR"]


def fold_mean(run: Path, tag: str, metric: str, model: str) -> float | None:
    path = run / f"folds_{tag}_{model}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return float(df[metric].mean()) if metric in df.columns else None


def gap(run: Path, metric: str, model: str) -> float | None:
    """CIC minus UNSW on the E2-restricted feature set."""
    a = fold_mean(run, "E2_CIC", metric, model)
    b = fold_mean(run, "E2_UNSW", metric, model)
    return None if a is None or b is None else a - b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--label-cic", default="Label")
    ap.add_argument("--label-unsw", default="label")
    ap.add_argument("--baseline", default="results/main",
                    help="directory holding the 12-pair E2 result to compare against")
    ap.add_argument("--outdir", default="results/e2_sensitivity")
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--model", default="XGB")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--script", default="src/ids_evaluation.py")
    ap.add_argument("--skip-runs", action="store_true",
                    help="only recompute the comparison from existing outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline)

    if gap(baseline, "balanced_accuracy", args.model) is None:
        raise SystemExit(f"no baseline E2 result found in {baseline} "
                         f"(expected folds_E2_CIC_{args.model}.csv)")

    for name, config, description in VARIANTS:
        run_dir = outdir / name
        if args.skip_runs and run_dir.exists():
            print(f"skipping {name} (--skip-runs)")
            continue
        print(f"\n=== {name}: {description}")
        cmd = [sys.executable, args.script,
               "--cic", args.cic, "--unsw", args.unsw,
               "--label-cic", args.label_cic, "--label-unsw", args.label_unsw,
               "--config", str(Path(args.config_dir) / config),
               "--outdir", str(run_dir),
               "--sample", str(args.sample),
               "--models", args.model,
               "--n-splits", "5", "--n-repeats", "3",
               "--skip-shap"]
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"run '{name}' failed with code {result.returncode}")

    # ---- comparison ----
    rows = []
    base_gaps = {m: gap(baseline, m, args.model) for m in METRICS}
    for metric in METRICS:
        row = {"metric": metric, "baseline_gap": base_gaps[metric]}
        for name, _, _ in VARIANTS:
            g = gap(outdir / name, metric, args.model)
            row[f"{name}_gap"] = g
            row[f"{name}_change"] = None if g is None else g - base_gaps[metric]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "e2_sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("E2 sensitivity: CIC-UNSW-NB15 minus UNSW-NB15, on the restricted set")
    print("=" * 78)
    show = df.copy()
    for c in show.columns:
        if c != "metric":
            show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:+.4f}")
    print(show.to_string(index=False))

    print("\nInterpretation")
    print("-" * 78)
    ba = df[df.metric == "balanced_accuracy"].iloc[0]
    for name, _, description in VARIANTS:
        change = ba.get(f"{name}_change")
        if pd.isna(change):
            continue
        pct = 100 * change / ba["baseline_gap"]
        verdict = ("holds" if abs(pct) < 20 else
                   "moves materially" if abs(pct) < 50 else "does not hold")
        print(f"{name:11s} balanced-accuracy gap {ba['baseline_gap']:+.4f} -> "
              f"{ba[f'{name}_gap']:+.4f}  ({pct:+.1f}%)  -> result {verdict}")
    print("""
If both variants leave the gap broadly intact, report that and remove the
sensitivity caveat from the limitations section.

If the no_window variant narrows the gap appreciably, that is a real finding
and should be reported as one: part of what the twelve-feature restriction was
crediting to the CICFlowMeter representation came from a pair whose two sides
are not strictly equivalent. Report the ten-pair figure as the primary E2
result and keep the twelve-pair figure for comparison.

If the swapped variant changes the gap appreciably, the two tools disagree on
flow direction often enough that the convention is not immaterial. That is a
finding about the datasets, not a defect in the analysis, and it belongs in
Section 4.5.""")
    print(f"\nwritten -> {outdir / 'e2_sensitivity.csv'}")


if __name__ == "__main__":
    main()
