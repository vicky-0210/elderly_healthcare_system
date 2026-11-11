"""Figures for the IC-BiMamba SisFall results (reads research/results/icbimamba_v1/<tag>/).

    .venv/Scripts/python modules/imu/experiments/icbimamba/figures.py --tag main
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("research/results/icbimamba_v1")
# Published SisFall results (percent). None = not reported.
PAPERS = [
    ("Sucerquia 2017", 96.1, None, None, None),
    ("CNN-LSTM, Sensors 2026", 98.89, None, 96.67, None),
    ("Zhang 2024, JMIR", 99.32, 98.58, 99.15, 98.86),
]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
OURS, PRIOR, PAPER = "#2a6fdb", "#9aa6b2", "#c9a227"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); args = ap.parse_args()
    out = RESULTS / args.tag
    s = json.loads((out / "summary.json").read_text()); e = s["ensemble"]
    cmp_ = pd.read_csv(out / "comparison.csv")
    oof = pd.read_csv(out / "oof_trials.csv")

    # 1. Confusion matrix
    tp = int(((oof.decision == 1) & (oof.y == 1)).sum()); tn = int(((oof.decision == 0) & (oof.y == 0)).sum())
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cells = [[tn, e["fp"]], [e["fn"], tp]]
    ax.imshow([[1, 0], [0, 1]], cmap="Blues", vmin=-.6, vmax=1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cells[i][j]:,}", ha="center", va="center", fontsize=15,
                    color="white" if i == j else INK)
    ax.set_xticks([0, 1], ["Predicted ADL", "Predicted fall"]); ax.set_yticks([0, 1], ["ADL", "Fall"])
    ax.set_title("IC-BiMamba ensemble, 4,500 SisFall trials", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=200); plt.close(fig)

    # 2. Comparison with papers and our earlier models
    rows = [(n, a, p, r, f) for n, a, p, r, f in PAPERS]
    names = {"xgb_native_100": "XGBoost (ours, earlier)", "dm_tcn_standard": "DM-TCN (ours, earlier)",
             "inception": "InceptionTime (ours, earlier)"}
    for k, v in names.items():
        b = cmp_[cmp_.model == k].iloc[0]
        rows.append((v, 100 * b.accuracy, 100 * b.precision, 100 * b.recall, 100 * b.f1))
    rows.append(("IC-BiMamba (ours)", 100 * e["accuracy"], 100 * e["precision"], 100 * e["recall"], 100 * e["f1"]))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for m, ax in enumerate(axes):
        for i, r in enumerate(rows):
            n, v = r[0], r[m + 1]
            if v is None:
                ax.text(95.6, i, "not reported", va="center", fontsize=8, color=MUTED); continue
            col = OURS if "IC-BiMamba" in n else PRIOR if "ours" in n else PAPER
            ax.barh(i, v, color=col, height=.6)
            ax.text(v + .03, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(95.5, 100.4); ax.set_title(metrics[m], color=INK, fontsize=10); style(ax)
        ax.xaxis.grid(True, color=GRID, lw=.6); ax.yaxis.grid(False)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows]); axes[0].set_ylim(len(rows) - .5, -.5)
    fig.suptitle("SisFall fall detection (%). Gold = published, grey = our earlier models, blue = IC-BiMamba",
                 fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(out / "comparison_bars.png", dpi=200); plt.close(fig)

    # 3. Errors by activity
    err = pd.DataFrame(s["errors_by_activity"]).sort_values("n")
    err["label"] = err.activity + err.y.map({1: " (missed fall)", 0: " (false alarm)"})
    fig, ax = plt.subplots(figsize=(6, .32 * len(err) + 1))
    ax.barh(err.label, err.n, color=[OURS if y else "#d1495b" for y in err.y], height=.6)
    for i, v in enumerate(err.n): ax.text(v + .1, i, str(v), va="center", fontsize=8, color=INK)
    ax.set_title(f"Errors by activity ({e['fn']} missed falls, {e['fp']} false alarms)", fontsize=10, color=INK)
    style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID, lw=.6)
    fig.tight_layout(); fig.savefig(out / "errors_by_activity.png", dpi=200); plt.close(fig)
    print("saved figures to", out)


if __name__ == "__main__":
    main()
