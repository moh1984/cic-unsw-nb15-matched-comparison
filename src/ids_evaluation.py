#!/usr/bin/env python3
"""
Reproducible evaluation pipeline for the comparative study of
CIC-UNSW-NB15 vs. UNSW-NB15.

Produces every number required by the revision guide:
  * dataset characterisation table
  * full metric set (accuracy, balanced accuracy, precision, recall,
    F1, macro F1, FAR, MCC, ROC-AUC, PR-AUC) as mean +/- SD over
    repeated stratified cross-validation
  * confusion matrices
  * computational cost (training time, per-flow inference latency)
  * McNemar pairwise significance tests with Holm correction
  * matched-condition experiments E1 (prior + size matched) and
    E2 (common feature subset)
  * feature importance / SHAP attribution

Usage:
    python ids_evaluation.py --cic CIC.csv --unsw UNSW.csv --outdir results/

Everything that must be checked by the authors is marked CHECK.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

RANDOM_STATE = 42          # overridden by --seed
BLOCK_SEED = 20260101      # fixed: defines the disjoint blocks, independent of --seed
N_SPLITS = 5
N_REPEATS = 3

# ---------------------------------------------------------------------------
# CHECK #1 -- THE MOST IMPORTANT LIST IN THIS FILE.
# Any identifier left in the feature matrix lets the model memorise which
# hosts generated the attacks instead of learning traffic behaviour. That
# single mistake is the usual cause of implausible >98% scores. Add every
# identifier-like column present in your own copies of the files.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = {
    # CICFlowMeter output
    "flow id", "flowid", "src ip", "source ip", "src port", "source port",
    "dst ip", "destination ip", "dst port", "destination port", "timestamp",
    # UNSW-NB15
    "srcip", "sport", "dstip", "dsport", "stime", "ltime",
    # generic
    "id", "no", "index", "unnamed: 0",
    # label-derived: attack_cat is a deterministic function of the binary label
    "attack_cat", "attack category",
}

# CHECK #2 -- semantic alignment for experiment E2. Left side = UNSW-NB15,
# right side = CIC-UNSW-NB15. Verify these against your actual headers; the
# CICFlowMeter column names vary between tool versions.
COMMON_FEATURE_MAP = {
    "dur": "Flow Duration",
    "proto": "Protocol",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "smeansz": "Fwd Packet Length Mean",
    "dmeansz": "Bwd Packet Length Mean",
    "sload": "Flow Bytes/s",
    "dload": "Flow Packets/s",
}


# ---------------------------------------------------------------------------
# loading and cleaning
# ---------------------------------------------------------------------------
def norm(name: str) -> str:
    return str(name).strip().lower()


def read_csv_sampled(path: str, sample: int | None, nrows: int | None,
                     chunksize: int = 500_000) -> tuple[pd.DataFrame, int]:
    """Read a CSV, optionally down-sampling at random without loading it whole.

    --nrows is a fast non-random head slice, for smoke-testing the pipeline only.
    --sample is a uniform random subsample, safe for the reported experiments.
    """
    if nrows:
        return pd.read_csv(path, low_memory=False, nrows=nrows), nrows
    if not sample:
        df = pd.read_csv(path, low_memory=False)
        return df, len(df)

    with open(path, "rb") as fh:            # cheap line count for the fraction
        total = sum(1 for _ in fh) - 1
    if total <= sample:
        df = pd.read_csv(path, low_memory=False)
        return df, len(df)

    frac = sample / total
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        mask = rng.random(len(chunk)) < frac
        parts.append(chunk[mask])
    print(f"  subsampled {sample:,} of {total:,} rows ({100 * frac:.2f}%)")
    return pd.concat(parts, ignore_index=True), total


def load_dataset(path: str, label_col: str | None, sample: int | None = None,
                 nrows: int | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load a CSV, drop leakage columns, clean invalid values, return X, y, report."""
    df, total_rows = read_csv_sampled(path, sample, nrows)
    df.columns = [str(c).strip() for c in df.columns]
    report: dict[str, object] = {"raw_rows_in_file": total_rows,
                                 "rows_loaded": len(df),
                                 "raw_features": df.shape[1] - 1}

    if label_col is None:
        candidates = [c for c in df.columns if norm(c) in {"label", "class", "attack"}]
        if not candidates:
            raise SystemExit(
                f"Could not infer the label column in {path}. "
                f"Pass it explicitly with --label-cic / --label-unsw. "
                f"Columns: {list(df.columns)[:25]}"
            )
        label_col = candidates[0]
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not in {path}")

    y_raw = df[label_col]
    df = df.drop(columns=[label_col])

    # binary label: 0 = benign, 1 = malicious
    # NB: pandas >= 3.0 gives text columns a dedicated 'str' dtype, not object,
    # so test for "not numeric" rather than "== object".
    if not pd.api.types.is_numeric_dtype(y_raw):
        y = (~y_raw.astype(str).str.strip().str.lower().isin(
            {"benign", "normal", "0", "no", "false"})).astype(int)
    else:
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)

    dropped = [c for c in df.columns if norm(c) in LEAKAGE_COLUMNS]
    df = df.drop(columns=dropped)
    report["dropped_identifier_columns"] = dropped

    # CICFlowMeter emits inf for zero-duration flows (Flow Bytes/s etc.)
    df = df.replace([np.inf, -np.inf], np.nan)
    n_inf_rows = int(df.isna().any(axis=1).sum())
    keep = df.notna().all(axis=1)
    df, y = df[keep], y[keep]
    report["rows_dropped_inf_nan"] = n_inf_rows

    before = len(df)
    dup = df.duplicated()
    df, y = df[~dup], y[~dup]
    report["rows_dropped_duplicate"] = before - len(df)

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    report["dropped_constant_columns"] = constant

    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)

    report.update(
        rows_used=len(df),
        features_used=df.shape[1],
        benign=int((y == 0).sum()),
        malicious=int((y == 1).sum()),
        benign_pct=round(100 * float((y == 0).mean()), 2),
        malicious_pct=round(100 * float((y == 1).mean()), 2),
    )
    return df, y, report


def disjoint_block(X: pd.DataFrame, y: pd.Series, n_blocks: int, block: int):
    """Keep one of n_blocks mutually disjoint stratified partitions of the data.

    Running the pipeline once per block gives estimates from non-overlapping
    records, so a confidence interval computed across blocks does not suffer
    the dependence that affects repeated cross-validation on a single sample.
    """
    if not n_blocks or n_blocks <= 1:
        return X, y
    if not 0 <= block < n_blocks:
        raise SystemExit(f"--block must be in [0, {n_blocks - 1}]")
    rng = np.random.default_rng(BLOCK_SEED)
    keep = np.zeros(len(y), dtype=bool)
    for label in (0, 1):                       # stratify so every block shares the prior
        idx = np.flatnonzero(y.to_numpy() == label)
        rng.shuffle(idx)
        keep[np.array_split(idx, n_blocks)[block]] = True
    print(f"  block {block + 1}/{n_blocks}: {int(keep.sum()):,} rows, "
          f"{100 * y[keep].mean():.2f}% attack")
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Min-Max on numeric, one-hot on categorical.

    Wrapped in a ColumnTransformer inside a Pipeline so that fitting happens
    on the training fold only -- this is what prevents scaling leakage.
    """
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=0.01, sparse_output=False), cat),
        ],
        remainder="drop",
    )


def classifiers(subset: str | None = None) -> dict:
    """CHECK #3 -- replace these with the values your own search selected,
    and report the search space and selected values in the paper."""
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
            eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE)
    if subset:
        want = [m.strip() for m in subset.split(",") if m.strip()]
        missing = [m for m in want if m not in models]
        if missing:
            raise SystemExit(f"Unknown model(s): {missing}. Available: {list(models)}")
        models = {k: models[k] for k in want}
    return models


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def score(y_true, y_pred, y_prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(y_true, y_pred, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, zero_division=0),
        "precision_benign": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_benign": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # FAR is the metric that decides whether a detector is deployable
        "FAR": fp / max(fp + tn, 1),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    return out


def cross_validate(X, y, model, n_splits=N_SPLITS, n_repeats=N_REPEATS) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        pipe = Pipeline([("prep", build_preprocessor(X)),
                         ("clf", model)])
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = pipe.predict(X.iloc[te])
        pred_s = time.perf_counter() - t0

        prob = None
        if hasattr(pipe, "predict_proba"):
            try:
                prob = pipe.predict_proba(X.iloc[te])[:, 1]
            except Exception:
                prob = None

        row = score(y.iloc[te], pred, prob)
        row.update(fold=fold, train_time_s=fit_s,
                   inference_us_per_flow=1e6 * pred_s / len(te),
                   throughput_flows_s=len(te) / max(pred_s, 1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> dict:
    num = folds.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
    out = {}
    for col in num.columns:
        out[f"{col}_mean"] = num[col].mean()
        out[f"{col}_sd"] = num[col].std(ddof=1)
        out[col] = f"{num[col].mean():.4f} ± {num[col].std(ddof=1):.4f}"
    return out


# ---------------------------------------------------------------------------
# significance testing
# ---------------------------------------------------------------------------
def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact / chi-square McNemar test on paired correctness vectors."""
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[0, n01], [n10, 0]]
        exact = (n01 + n10) < 25
        p = float(mcnemar(table, exact=exact, correction=not exact).pvalue)
    except ImportError:
        from scipy.stats import binomtest
        n = n01 + n10
        p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return n01, n10, p


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def pairwise_significance(X, y, models: dict, outdir: Path, tag: str) -> pd.DataFrame:
    """Fit every model on one common split, then compare on identical instances.
    McNemar requires paired predictions, hence the shared holdout."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    correctness, preds = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", model)])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        preds[name] = p
        correctness[name] = (p == y_te.to_numpy())
        pd.DataFrame(confusion_matrix(y_te, p, labels=[0, 1]),
                     index=["true_benign", "true_attack"],
                     columns=["pred_benign", "pred_attack"]
                     ).to_csv(outdir / f"confusion_{tag}_{name}.csv")

    rows, raw_p = [], []
    for a, b in combinations(models, 2):
        n01, n10, p = mcnemar_pvalue(correctness[a], correctness[b])
        rows.append({"model_a": a, "model_b": b,
                     "a_wrong_b_right": n01, "a_right_b_wrong": n10,
                     "p_value": p})
        raw_p.append(p)
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(raw_p)
        df["significant_a005"] = df["p_holm"] < 0.05
    return df


# ---------------------------------------------------------------------------
# matched-condition experiments
# ---------------------------------------------------------------------------
def feasible_n(y, target_attack_ratio: float) -> int:
    """Largest sample size at which a dataset can reach the target attack ratio.

    Bounded by whichever class runs out first. Without this, a dataset with few
    benign records silently ends up smaller than the one it is compared against,
    which would leave training-set size uncontrolled in E1.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return int(min(n_pos / max(target_attack_ratio, 1e-9),
                   n_neg / max(1 - target_attack_ratio, 1e-9)))


def match_prior_and_size(X, y, target_attack_ratio: float, target_n: int,
                         seed: int = RANDOM_STATE):
    """Stratified subsample to a given attack ratio and total size (experiment E1)."""
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y.to_numpy() == 1)
    idx_neg = np.flatnonzero(y.to_numpy() == 0)
    n_pos = int(round(target_n * target_attack_ratio))
    n_neg = target_n - n_pos
    if n_pos > len(idx_pos) or n_neg > len(idx_neg):
        scale = min(len(idx_pos) / max(n_pos, 1), len(idx_neg) / max(n_neg, 1))
        n_pos, n_neg = int(n_pos * scale), int(n_neg * scale)
    sel = np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                          rng.choice(idx_neg, n_neg, replace=False)])
    rng.shuffle(sel)
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)


def common_feature_view(X_unsw, X_cic):
    """Experiment E2: restrict both datasets to a semantically aligned subset."""
    lo_u = {norm(c): c for c in X_unsw.columns}
    lo_c = {norm(c): c for c in X_cic.columns}
    pairs = [(lo_u[norm(u)], lo_c[norm(c)])
             for u, c in COMMON_FEATURE_MAP.items()
             if norm(u) in lo_u and norm(c) in lo_c]
    if not pairs:
        return None, None, []
    u_cols = [p[0] for p in pairs]
    c_cols = [p[1] for p in pairs]
    a = X_unsw[u_cols].copy()
    b = X_cic[c_cols].copy()
    canonical = [f"f{i}" for i in range(len(pairs))]
    a.columns = canonical
    b.columns = canonical
    return a, b, pairs


# ---------------------------------------------------------------------------
# feature attribution
# ---------------------------------------------------------------------------
def feature_attribution(X, y, model, name: str, tag: str, outdir: Path,
                        sample: int = 20000):
    pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", model)])
    idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X), min(sample, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    pipe.fit(Xs, ys)
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(Xs.shape[1])]
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        pd.DataFrame({"feature": names, "importance": clf.feature_importances_}) \
            .sort_values("importance", ascending=False) \
            .to_csv(outdir / f"feature_importance_{tag}_{name}.csv", index=False)
    try:  # optional: SHAP gives signed, per-instance attribution
        import shap
        expl = shap.TreeExplainer(clf)
        Xt = pipe.named_steps["prep"].transform(Xs[:2000])
        vals = expl.shap_values(Xt)
        vals = vals[1] if isinstance(vals, list) else vals
        pd.DataFrame({"feature": names,
                      "mean_abs_shap": np.abs(vals).mean(axis=0)}) \
            .sort_values("mean_abs_shap", ascending=False) \
            .to_csv(outdir / f"shap_{tag}_{name}.csv", index=False)
    except Exception as exc:
        print(f"  [shap skipped for {name}: {exc}]")


# ---------------------------------------------------------------------------
DISPLAY_COLS = ["dataset", "model", "accuracy", "balanced_accuracy",
                "precision_attack", "recall_attack", "f1_attack", "macro_f1",
                "FAR", "MCC", "roc_auc", "pr_auc"]


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns that go into the paper's tables."""
    return df[[c for c in DISPLAY_COLS if c in df.columns]]


def evaluate_all(X, y, tag: str, outdir: Path, cfg) -> pd.DataFrame:
    rows = []
    for name, model in classifiers(cfg.models).items():
        print(f"  {tag}: {name}")
        folds = cross_validate(X, y, model, cfg.n_splits, cfg.n_repeats)
        folds.to_csv(outdir / f"folds_{tag}_{name}.csv", index=False)
        row = {"dataset": tag, "model": name}
        row.update(summarise(folds))
        rows.append(row)
    df = pd.DataFrame(rows)
    tidy(df).to_csv(outdir / f"main_results_{tag}.csv", index=False)
    df[[c for c in ["dataset", "model", "train_time_s",
                    "inference_us_per_flow", "throughput_flows_s"]
        if c in df.columns]].to_csv(outdir / f"cost_{tag}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--label-cic", default=None)
    ap.add_argument("--label-unsw", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sample", type=int, default=None,
                    help="random stratified subsample size per dataset "
                         "(recommended on Colab: 200000-500000)")
    ap.add_argument("--nrows", type=int, default=None,
                    help="read only the first N rows -- SMOKE TEST ONLY, not random")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset, e.g. RF,XGB,DT "
                         "(KNN is impractical above ~300k rows)")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--skip-matched", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for sampling, splitting and the classifiers")
    ap.add_argument("--disjoint-blocks", type=int, default=None,
                    help="partition the data into N disjoint stratified blocks "
                         "and use only one of them (see --block); run once per "
                         "block to obtain independent estimates")
    ap.add_argument("--block", type=int, default=0,
                    help="which block to use, 0-indexed")
    ap.add_argument("--config", default=None,
                    help="JSON file with optional keys 'extra_leakage_columns' "
                         "(list, appended) and 'common_feature_map' (dict, replaces)")
    args = ap.parse_args()

    global RANDOM_STATE
    RANDOM_STATE = args.seed
    if args.seed != 42:
        print(f"seed set to {RANDOM_STATE}")

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for col in cfg.get("extra_leakage_columns", []):
            LEAKAGE_COLUMNS.add(norm(col))
        if "common_feature_map" in cfg:
            COMMON_FEATURE_MAP.clear()
            COMMON_FEATURE_MAP.update(cfg["common_feature_map"])
        print(f"config applied: {len(LEAKAGE_COLUMNS)} leakage columns, "
              f"{len(COMMON_FEATURE_MAP)} mapped feature pairs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("loading CIC-UNSW-NB15 ...")
    Xc, yc, rc = load_dataset(args.cic, args.label_cic, args.sample, args.nrows)
    print("loading UNSW-NB15 ...")
    Xu, yu, ru = load_dataset(args.unsw, args.label_unsw, args.sample, args.nrows)

    if args.disjoint_blocks:
        print(f"\nrestricting to disjoint block {args.block + 1} of {args.disjoint_blocks}")
        Xc, yc = disjoint_block(Xc, yc, args.disjoint_blocks, args.block)
        Xu, yu = disjoint_block(Xu, yu, args.disjoint_blocks, args.block)
        rc["disjoint_block"] = ru["disjoint_block"] = f"{args.block + 1}/{args.disjoint_blocks}"
        rc["rows_used"], ru["rows_used"] = len(Xc), len(Xu)

    pd.DataFrame({"CIC-UNSW-NB15": rc, "UNSW-NB15": ru}).to_csv(
        outdir / "dataset_summary.csv")
    print(json.dumps({"CIC": rc, "UNSW": ru}, indent=2, default=str))

    print("\nbaseline evaluation (datasets as published)")
    base = pd.concat([evaluate_all(Xc, yc, "CIC", outdir, args),
                      evaluate_all(Xu, yu, "UNSW", outdir, args)])
    base.to_csv(outdir / "baseline_all_columns.csv", index=False)

    print("\nMcNemar pairwise tests")
    for tag, X, y in (("CIC", Xc, yc), ("UNSW", Xu, yu)):
        pairwise_significance(X, y, classifiers(args.models), outdir, tag).to_csv(
            outdir / f"mcnemar_{tag}.csv", index=False)

    if args.skip_matched:
        print("\nmatched-condition experiments skipped (--skip-matched)")
        print(f"\ndone -> {outdir.resolve()}")
        return

    print("\nE1: prior- and size-matched comparison")
    ratio = float((yc == 1).mean())          # CIC ratio is the target
    # both datasets must be able to *reach* this ratio at this size, otherwise
    # size stays uncontrolled and E1 answers a different question
    n = min(feasible_n(yc, ratio), feasible_n(yu, ratio), len(Xc), len(Xu))
    Xc1, yc1 = match_prior_and_size(Xc, yc, ratio, n)
    Xu1, yu1 = match_prior_and_size(Xu, yu, ratio, n)
    print(f"  target attack ratio={ratio:.4f}  matched n={n:,} per dataset")
    print(f"  CIC:  {len(Xc1):,} rows, {100 * yc1.mean():.2f}% attack")
    print(f"  UNSW: {len(Xu1):,} rows, {100 * yu1.mean():.2f}% attack")
    tidy(pd.concat([evaluate_all(Xc1, yc1, "E1_CIC", outdir, args),
                    evaluate_all(Xu1, yu1, "E1_UNSW", outdir, args)])).to_csv(
        outdir / "matched_E1.csv", index=False)

    print("\nE2: common feature subset")
    Au, Bc, pairs = common_feature_view(Xu1, Xc1)
    if pairs:
        pd.DataFrame(pairs, columns=["unsw_column", "cic_column"]).to_csv(
            outdir / "E2_feature_mapping.csv", index=False)
        tidy(pd.concat([evaluate_all(Bc, yc1, "E2_CIC", outdir, args),
                        evaluate_all(Au, yu1, "E2_UNSW", outdir, args)])).to_csv(
            outdir / "matched_E2.csv", index=False)
    else:
        print("  no columns matched -- edit COMMON_FEATURE_MAP (CHECK #2) "
              "to reflect your actual headers, then rerun.")

    if not args.skip_shap:
        print("\nfeature attribution")
        models = classifiers(args.models)
        best = "XGB" if "XGB" in models else list(models)[0]
        feature_attribution(Xc, yc, models[best], best, "CIC", outdir)
        feature_attribution(Xu, yu, models[best], best, "UNSW", outdir)

    print(f"\ndone -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
