"""Secondary matched alert-budget analysis, using the same policies for every encoder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.datasets.osasud import RUN
from wearable_ppg.evaluation.binary_study import NAMES, REFERENCES, event_counts, read_binary
from wearable_ppg.training.dual_rate import REPORT_DIR, RUN_DIR


def trailing_mean(data, n=5):
    p = data["p"].astype(np.float64)
    cuts = np.flatnonzero((np.diff(data["t"]) != 1) | (np.diff(data["patient"]) != 0)) + 1
    smoothed = np.empty_like(p)
    for part in np.split(np.arange(len(p)), cuts):
        cumul = np.r_[0., np.cumsum(p[part])]
        end = np.arange(1, len(part)+1)
        start = np.maximum(0, end-n)
        smoothed[part] = (cumul[end] - cumul[start]) / (end-start)
    return smoothed


def policies(data, p):
    # Extend the original 26 event-F1 thresholds into the upper tail for the alarm budgets.
    quantiles = np.unique(np.r_[np.linspace(.40, .99, 26), .90, .95, .975, .995, .9975, .999, .9995, .9999])
    points = []
    for threshold in np.r_[np.unique(np.quantile(p, quantiles)), 1.01]:
        (tp, fp, _), rows, _ = event_counts(data, p>=threshold)
        points.append((threshold, tp, fp, np.mean([r["event_f1"] for r in rows if r["true_events"]])))
    result = {"event_f1": max(points, key=lambda r: (r[3],r[0]))}
    hours = len(p)/3600
    for budget in (1, 10, 30):
        result[f"fp{budget}"] = max((r for r in points if r[2] <= hours*budget), key=lambda r: (r[1],-r[2],r[0]))
    return result


def plot_budgets(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("CNN + Transformer", "TCN + GRU", "Binary control", "Dual, hard labels",
              "Dual, fraction labels", "Dual, auxiliary fraction")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, layout="constrained")
    for ax, mode, title in zip(axes, ("raw", "mean5"), ("Per-second scores", "Trailing five-second average")):
        for model, label in zip(NAMES, labels):
            curve = table[(table.model == model) & (table["filter"] == mode)].set_index("policy")
            points = curve.loc[["fp1", "fp10", "fp30"]]
            ax.plot(points.false_per_h, 100*points.event_recall, marker="o", markersize=4, label=label)
        ax.set(title=title, xlabel="Held-out false events per scored hour", xlim=(0, None), ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out reference events matched (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Operating points tuned for 1, 10 and 30 false events per validation hour")
    fig.savefig(REPORT_DIR / "alert_budgets.png", dpi=160)
    plt.close(fig)


def main():
    records = []
    for name in NAMES:
        for k in range(5):
            folder = (RUN if name in REFERENCES else RUN_DIR) / name / f"fold{k}"
            val, test = (read_binary(folder,s) for s in ("validation", "test"))
            for mode in ("raw", "mean5"):
                pv, pt = ((val["p"], test["p"]) if mode == "raw" else (trailing_mean(val), trailing_mean(test)))
                for rule, (thr, vt, vf, _) in policies(val, pv).items():
                    (tp, fp, fn), _, _ = event_counts(test, pt >= thr)
                    records.append(dict(model=name, fold=k, filter=mode, policy=rule, threshold=thr,
                                        val_matches=vt, val_fp_per_h=vf*3600/len(pv),
                                        tp=tp, fp=fp, fn=fn, hours=len(pt)/3600))
    by_fold = pd.DataFrame(records)
    by_fold.to_csv(REPORT_DIR / "alert_policy_folds.csv", index=False)
    pooled = by_fold.groupby(["model", "filter", "policy"], as_index=False)[["tp", "fp", "fn", "hours"]].sum()
    pooled["event_recall"] = pooled.tp/(pooled.tp+pooled.fn)
    pooled["event_f1"] = 2*pooled.tp/(2*pooled.tp+pooled.fp+pooled.fn)
    pooled["false_per_h"] = pooled.fp/pooled.hours
    pooled.to_csv(REPORT_DIR / "alert_policies.csv", index=False)
    plot_budgets(pooled)
    print(pooled[pooled.policy == "fp10"][["model","filter","event_recall","false_per_h"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
