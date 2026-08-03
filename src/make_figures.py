#!/usr/bin/env python3
"""Generate publication figures for the CIC-UNSW-NB15 vs UNSW-NB15 paper.

Reads only the CSVs written by ids_evaluation.py, so every value plotted is
traceable to a run. Error bars are standard deviations over the 15 folds
(3 x 5-fold stratified CV), computed from the folds_*.csv files rather than
re-derived, so the figures and the tables cannot disagree.

Usage:  python make_figures.py --results results/main --outdir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# colourblind-safe (Okabe-Ito)
C_CIC, C_UNSW = "#0072B2", "#D55E00"
MODELS = ["XGB", "RF", "AdaBoost", "DT", "KNN"]


def folds(res: Path, tag: str, model: str) -> pd.DataFrame | None:
    p = res / f"folds_{tag}_{model}.csv"
    return pd.read_csv(p) if p.exists() else None


def stat(res: Path, tag: str, model: str, metric: str) -> tuple[float, float]:
    df = folds(res, tag, model)
    if df is None or metric not in df:
        return np.nan, np.nan
    return float(df[metric].mean()), float(df[metric].std(ddof=1))


# ---------------------------------------------------------------- figure 2
def fig_accuracy(res: Path, out: Path) -> None:
    """Per-model accuracy on both datasets, with SD error bars."""
    x = np.arange(len(MODELS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    for off, tag, colour, label in ((-w / 2, "CIC", C_CIC, "CIC-UNSW-NB15"),
                                    (w / 2, "UNSW", C_UNSW, "UNSW-NB15")):
        m = [stat(res, tag, k, "accuracy")[0] for k in MODELS]
        s = [stat(res, tag, k, "accuracy")[1] for k in MODELS]
        bars = ax.bar(x + off, m, w, yerr=s, capsize=2.5, color=colour,
                      label=label, error_kw={"elinewidth": 0.8})
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{100 * v:.2f}",
                    ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.82, 1.005)
    ax.set_title("Classifier accuracy by dataset (mean ± SD, 3 × 5-fold CV)")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_accuracy.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_metrics(res: Path, out: Path) -> None:
    """Four imbalance-robust metrics side by side, best model per dataset."""
    metrics = [("balanced_accuracy", "Balanced accuracy"),
               ("macro_f1", "Macro F1"), ("MCC", "MCC"),
               ("recall_attack", "Recall (attack)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    for ax, (key, title) in zip(axes, metrics):
        x = np.arange(len(MODELS)); w = 0.36
        for off, tag, colour in ((-w / 2, "CIC", C_CIC), (w / 2, "UNSW", C_UNSW)):
            m = [stat(res, tag, k, key)[0] for k in MODELS]
            s = [stat(res, tag, k, key)[1] for k in MODELS]
            ax.bar(x + off, m, w, yerr=s, capsize=1.8, color=colour,
                   error_kw={"elinewidth": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0.70, 1.0)
    axes[0].set_ylabel("Score")
    fig.legend(["CIC-UNSW-NB15", "UNSW-NB15"], loc="lower center",
               ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig3_metrics.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_matched(res: Path, out: Path, model: str = "XGB") -> None:
    """The central result: accuracy converges while detection diverges."""
    stages = [("", "Baseline"), ("E1_", "E1: matched prior\n& size"),
              ("E2_", "E2: common\nfeatures")]
    panels = [("accuracy", "Accuracy"),
              ("balanced_accuracy", "Balanced accuracy"),
              ("recall_attack", "Recall (attack)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        cic = [stat(res, f"{p}CIC", model, key) for p, _ in stages]
        uns = [stat(res, f"{p}UNSW", model, key) for p, _ in stages]
        x = np.arange(len(stages))
        ax.errorbar(x, [c[0] for c in cic], yerr=[c[1] for c in cic],
                    marker="o", ms=5, lw=1.6, color=C_CIC, capsize=3,
                    label="CIC-UNSW-NB15")
        ax.errorbar(x, [u[0] for u in uns], yerr=[u[1] for u in uns],
                    marker="s", ms=5, lw=1.6, color=C_UNSW, capsize=3,
                    label="UNSW-NB15")
        for i in x:      # annotate the gap, which is the point of the figure
            gap = 100 * (cic[i][0] - uns[i][0])
            ymid = (cic[i][0] + uns[i][0]) / 2
            ax.annotate("", xy=(i, cic[i][0]), xytext=(i, uns[i][0]),
                        arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
            ax.text(i + 0.08, ymid, f"{gap:+.2f} pp", fontsize=6.8, color="0.25")
        ax.set_xticks(x); ax.set_xticklabels([s for _, s in stages], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.35, len(stages) - 0.4)
    axes[0].set_ylabel(f"Score ({model})")
    axes[0].legend(loc="center left", fontsize=7.5)
    fig.suptitle("Controlling class prior narrows the accuracy gap "
                 "but widens the detection gap", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig5_matched_conditions.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_far(res: Path, out: Path, model: str = "XGB") -> None:
    """FAR: the apparent threefold advantage is a prior artefact."""
    stages = [("", "Baseline"), ("E1_", "E1: matched"), ("E2_", "E2: common")]
    x = np.arange(len(stages)); w = 0.36
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for off, pre, colour, label in ((-w / 2, "CIC", C_CIC, "CIC-UNSW-NB15"),
                                    (w / 2, "UNSW", C_UNSW, "UNSW-NB15")):
        m = [100 * stat(res, f"{p}{pre}", model, "FAR")[0] for p, _ in stages]
        s = [100 * stat(res, f"{p}{pre}", model, "FAR")[1] for p, _ in stages]
        bars = ax.bar(x + off, m, w, yerr=s, capsize=2.5, color=colour,
                      label=label, error_kw={"elinewidth": 0.8})
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([s for _, s in stages], fontsize=8)
    ax.set_ylabel("False alarm rate (%)")
    ax.set_title(f"False alarm rate ({model})", fontsize=9.5)
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(out / "fig6_far.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_cost(res: Path, out: Path) -> None:
    """Accuracy against throughput: KNN is undeployable at any accuracy."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for tag, colour, marker, label in (("CIC", C_CIC, "o", "CIC-UNSW-NB15"),
                                       ("UNSW", C_UNSW, "s", "UNSW-NB15")):
        xs, ys = [], []
        for k in MODELS:
            thr, _ = stat(res, tag, k, "throughput_flows_s")
            acc, _ = stat(res, tag, k, "accuracy")
            xs.append(thr); ys.append(acc)
            ax.annotate(k, (thr, acc), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=colour)
        ax.scatter(xs, ys, s=42, color=colour, marker=marker, label=label,
                   zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvspan(1e2, 1e4, color="0.85", alpha=0.5, zorder=0)
    ax.text(1.6e3, 0.835, "impractical for\ninline deployment",
            fontsize=6.8, color="0.35", ha="center")
    ax.set_xlabel("Inference throughput (flows/s, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection quality against inference cost", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(out / "fig8_cost.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 7
def fig_confusion(res: Path, out: Path, model: str = "XGB") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    for ax, (tag, name) in zip(axes, (("CIC", "CIC-UNSW-NB15"),
                                      ("UNSW", "UNSW-NB15"))):
        p = res / f"confusion_{tag}_{model}.csv"
        if not p.exists():
            continue
        cm = pd.read_csv(p, index_col=0).to_numpy()
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if pct[i, j] > 55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. benign", "Pred. attack"],
                                                  fontsize=7.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True benign", "True attack"],
                                                  fontsize=7.5)
        ax.set_title(f"{name} ({model})", fontsize=8.5)
        ax.grid(False)
    fig.suptitle("Confusion matrices, held-out partition (row-normalised)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig4_confusion.png"); plt.close(fig)


# ---------------------------------------------------------------- figure 8
def fig_importance(res: Path, out: Path, top: int = 12) -> None:
    """SHAP where available, gain-based importance otherwise."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (tag, name, colour) in zip(axes,
                                       (("CIC", "CIC-UNSW-NB15", C_CIC),
                                        ("UNSW", "UNSW-NB15", C_UNSW))):
        shap_p = res / f"shap_{tag}_XGB.csv"
        imp_p = res / f"feature_importance_{tag}_XGB.csv"
        if shap_p.exists():
            df = pd.read_csv(shap_p); col, xlabel = "mean_abs_shap", "Mean |SHAP|"
        else:
            df = pd.read_csv(imp_p); col, xlabel = "importance", "Gain importance"
        df = df.nlargest(top, col).iloc[::-1]
        names = [n.replace("num__", "").replace("cat__", "") for n in df["feature"]]
        ax.barh(range(len(df)), df[col], color=colour, height=0.72)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(names, fontsize=6.8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(name, fontsize=8.5)
    fig.suptitle("Feature attribution for XGBoost", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig7_feature_attribution.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fig_accuracy(res, out)        # Figure 2
    fig_metrics(res, out)         # Figure 3
    fig_confusion(res, out)       # Figure 4
    fig_matched(res, out)         # Figure 5
    fig_far(res, out)             # Figure 6
    fig_importance(res, out)      # Figure 7
    fig_cost(res, out)            # Figure 8
    for f in sorted(out.glob("*.png")):
        print("wrote", f)


if __name__ == "__main__":
    main()
