import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(6.6, 8.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 25); ax.axis("off")

BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#F0F0F0", "#009E73"

def box(x, y, w, h, text, fc="white", ec="0.3", bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color="0.35"))

box(0.3, 22.6, 4.3, 1.5, "UNSW-NB15\n(Argus / Bro-IDS, 49 features)", fc="#DEEBF7", bold=True)
box(5.4, 22.6, 4.3, 1.5, "CIC-UNSW-NB15\n(CICFlowMeter)", fc="#FDE3D0", bold=True)
ax.text(5.0, 24.5, "same raw traffic capture", ha="center", fontsize=7.5,
        style="italic", color="0.4")

arrow(2.45, 22.6, 2.45, 21.6); arrow(7.55, 22.6, 7.55, 21.6)
box(1.3, 19.6, 7.4, 2.0,
    "Preprocessing\ndrop identifiers and label-derived attributes  •  replace Inf with NaN\n"
    "remove duplicates and constant columns  •  one-hot encode nominal attributes",
    fc=GREY)

arrow(5.0, 19.6, 5.0, 18.6)
box(0.6, 16.6, 8.8, 2.0,
    "Two evaluation paths, both with the Min–Max scaler fitted on training data only\n"
    "(A)  3 × 5-fold stratified CV over the whole sample  →  all reported metrics, mean ± SD\n"
    "(B)  one stratified 70/30 split  →  paired predictions for McNemar, confusion matrices",
    fc="#FFF6E5", fs=7.8)

arrow(5.0, 16.6, 5.0, 16.1)
box(1.5, 14.6, 7.0, 1.5,
    "Five classifiers: RF, XGBoost, AdaBoost, Decision Tree, KNN\n"
    "fixed configurations (Table 3); no hyper-parameter search performed", bold=True)

arrow(3.0, 14.6, 3.0, 13.6); arrow(7.0, 14.6, 7.0, 13.6)
box(0.3, 11.8, 4.6, 1.8,
    "Detection metrics\nAccuracy, Balanced Accuracy\nPrecision, Recall, F1, Macro F1\n"
    "FAR, MCC, ROC-AUC, PR-AUC", fs=7.5)
box(5.1, 11.8, 4.6, 1.8,
    "Cost metrics\nTraining time\nInference latency per flow\nThroughput", fs=7.5)

arrow(2.6, 11.8, 3.6, 10.8); arrow(7.4, 11.8, 6.4, 10.8)
box(0.8, 8.3, 8.4, 2.5,
    "Matched-condition protocol\n"
    "E1  equalise class prior and training-set size across the two datasets\n"
    "E2  restrict both datasets to 12 semantically aligned features (Table 4)\n"
    "E3  paired flows — not feasible: five-tuple removed from both public releases",
    fc="#E2F0E8", ec=GREEN, bold=True)

arrow(5.0, 8.3, 5.0, 7.4)
box(1.3, 5.8, 7.4, 1.5,
    "Statistical validation\nMcNemar test on paired predictions with Holm correction  •  mean ± SD over 15 folds")

arrow(5.0, 5.6, 5.0, 4.8)
box(1.3, 3.3, 7.4, 1.5,
    "Ablation and attribution\nSHAP attribution  •  removal of the highest-importance attributes")

arrow(5.0, 3.3, 5.0, 2.3)
box(2.7, 0.9, 4.6, 1.4, "Comparative analysis", fc="#DEEBF7", bold=True, fs=9)

fig.tight_layout()
fig.savefig("fig1_methodology.png", bbox_inches="tight")
print("ok")
