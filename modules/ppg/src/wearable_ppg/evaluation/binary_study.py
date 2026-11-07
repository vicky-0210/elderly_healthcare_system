"""Evaluate binary follow-up candidates at aligned timestamps and validation-set operating points."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from wearable_ppg.datasets.osasud import RESULT, RUN
from wearable_ppg.evaluation.compare import choose, join, read_scores
from wearable_ppg.evaluation.metrics import event_rows
from wearable_ppg.models.dual_rate import VARIANTS
from wearable_ppg.training.dual_rate import REPORT_DIR, RUN_DIR


REFERENCES = ("cnn_transformer", "tcn_gru")
NAMES = REFERENCES + VARIANTS
KEYS = ("y", "p", "t", "patient", "arrival", "decision")


def read_binary(path, split):
    data = read_scores(path, split)
    return {key: data[key] for key in KEYS}


def check_alignment(data, reference, group):
    assert set(data["patient"]) == set(group)
    for key in ("y", "t", "patient", "arrival"):
        np.testing.assert_array_equal(data[key], reference[key])
    assert np.isfinite(data["p"]).all() and np.all((data["p"] >= 0) & (data["p"] <= 1))


def event_counts(data, decision):
    rows, delay = event_rows(data["y"], decision, data["t"], data["patient"], data["arrival"])
    totals = np.asarray([[r[k] for k in ("event_tp", "event_fp", "event_fn")] for r in rows]).sum(axis=0)
    return totals, rows, delay


def low_burden_threshold(val):
    quantiles = [.9, .95, .975, .99, .995, .9975, .999, .9995, .9999]
    hours = len(val["p"])/3600
    choices = []
    for threshold in np.r_[np.quantile(val["p"], quantiles), 1.01]:
        (tp, fp, _), _, _ = event_counts(val, val["p"] >= threshold)
        if fp / hours <= 1:
            choices.append((tp, -fp, threshold))
    return float(max(choices)[2])


def summarize(name, data):
    y = data["y"] > 0
    (tp, fp, fn), people, delay = event_counts(data, data["decision"])
    (ltp, lfp, lfn), _, _ = event_counts(data, data["low_decision"])
    tn, sec_fp, sec_fn, sec_tp = confusion_matrix(y, data["decision"], labels=[False, True]).ravel()
    for row in people:
        mask = data["patient"] == row["patient"]
        row.update(model=name, ap=average_precision_score(y[mask], data["p"][mask]), prevalence=y[mask].mean())
    result = dict(model=name, patients=len(people), seconds=len(y), prevalence=y.mean(),
                  patient_ap=np.mean([r["ap"] for r in people]), pooled_ap=average_precision_score(y, data["p"]),
                  auroc=roc_auc_score(y, data["p"]), brier=brier_score_loss(y, data["p"]),
                  second_precision=sec_tp/max(1,sec_tp+sec_fp), second_recall=sec_tp/max(1,sec_tp+sec_fn),
                  specificity=tn/(tn+sec_fp), binary_f1=2*sec_tp/max(1,2*sec_tp+sec_fp+sec_fn),
                  event_tp=tp, event_fp=fp, event_fn=fn, event_precision=tp/max(1,tp+fp),
                  event_recall=tp/(tp+fn), event_f1=2*tp/max(1,2*tp+fp+fn),
                  false_per_h=fp/(len(y)/3600), matched_delay_median_s=np.median(delay) if delay else np.nan,
                  low_tp=ltp, low_fp=lfp, low_fn=lfn, low_recall=ltp/(ltp+lfn), low_false_per_h=lfp/(len(y)/3600))
    return result, people


def paired_intervals(people):
    pairs = [("dual_hard", "tcn_gru"), ("dual_hard", "binary_tcn_gru"), ("dual_hard", "cnn_transformer"),
             ("dual_fraction", "dual_hard"), ("dual_aux", "dual_hard"),
             ("nested_dual", "tcn_gru"), ("nested_dual", "cnn_transformer")]
    def draws(name):
        table = people[people.model == name].sort_values("patient")
        a = table[["ap", "event_tp", "event_fp", "event_fn", "hours"]].to_numpy()[idx]
        ap, tp, fp, fn, hours = a.sum(axis=1).T
        return np.column_stack((ap/size, 2*tp/(2*tp+fn+fp), fp/hours))
    size = people.patient.nunique()
    rng = np.random.default_rng(831)
    idx = np.vstack((np.arange(size), rng.integers(size, size=(3000, size))))
    out = []
    for left, right in pairs:
        differences = draws(left)-draws(right)
        for i, key in enumerate(("patient_ap", "event_f1", "false_per_h")):
            lower, upper = np.quantile(differences[1:, i], [.025, .975])
            out.append(dict(model=left, reference=right, metric=key, difference=differences[0, i],
                            lower_95=lower, upper_95=upper))
    return pd.DataFrame(out)


def evaluate():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULT / "folds.csv")
    held = {name: [] for name in NAMES}
    folds = []
    for k in range(5):
        ref_path = RUN / "tcn_gru" / f"fold{k}"
        ref = {split: read_binary(ref_path, split) for split in ("validation", "test")}
        for name in NAMES:
            root = RUN if name in REFERENCES else RUN_DIR
            path = root / name / f"fold{k}"
            meta = json.loads((path / "done.json").read_text())
            test, val = (read_binary(path, split) for split in ("test", "validation"))
            for rows, split, group, fold in ((test, "test", "test", k), (val, "validation", "val", (k+1)%5)):
                expected = set(manifest.loc[manifest.fold == fold, "patient"])
                assert set(meta[group]) == expected
                check_alignment(rows, ref[split], expected)
                np.testing.assert_array_equal(rows["decision"], rows["p"] >= meta["threshold"])
            assert set(meta["train"]) == set(manifest.patient) - set(meta["test"]) - set(meta["val"])
            low = low_burden_threshold(val)
            test["low_decision"] = test["p"] >= low
            held[name].append(test)
            _, vpatients, _ = event_counts(val, val["decision"])
            val_f1 = np.mean([r["event_f1"] for r in vpatients if r["true_events"] > 0])
            row, _ = summarize(name, test)
            folds.append(dict(fold=k, val_ap=meta["val_patient_ap"], val_event_f1=val_f1, threshold=meta["threshold"],
                              low_threshold=low, parameters=meta["parameters"], epoch=meta["epoch"], **row))
    by_fold = pd.DataFrame(folds)
    columns = ["val_ap", "val_event_f1", "parameters"]
    averages = by_fold.groupby("model", as_index=False)[columns].mean()
    candidates = list(VARIANTS[1:])
    selected = choose(averages[averages.model.isin(candidates)])
    per_fold = [choose(by_fold[(by_fold.fold == k) & by_fold.model.isin(candidates)]) for k in range(5)]
    held["nested_dual"] = [held[model][i] for i, model in enumerate(per_fold)]
    outcomes, patients = [], []
    for name, parts in held.items():
        results, people = summarize(name, join(parts))
        assert results["seconds"] == manifest.scored_seconds.sum()
        outcomes.append(results)
        patients.extend(people)
    table = pd.DataFrame(outcomes).merge(averages, on="model", how="left")
    people = pd.DataFrame(patients)
    table.to_csv(REPORT_DIR / "comparison.csv", index=False)
    by_fold.to_csv(REPORT_DIR / "folds.csv", index=False)
    people.to_csv(RUN_DIR / "patient_metrics.csv", index=False)
    paired_intervals(people).to_csv(REPORT_DIR / "paired.csv", index=False)
    info = dict(dual_candidate=selected, nested_dual_choices=per_fold,
                overall_validation_choice=choose(averages),
                rule="Highest validation macro-patient AP, within .005 prefer validation patient event F1 then fewer parameters")
    (REPORT_DIR / "selection.json").write_text(json.dumps(info, indent=2))
    plot(table, people)
    print(table[["model", "val_ap", "patient_ap", "event_f1", "event_recall", "false_per_h", "low_recall", "low_false_per_h"]].round(4).to_string(index=False))
    print(info)


def plot(table, people):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["CNN + Transformer", "TCN + GRU", "Binary control", "Dual, hard labels",
              "Dual, fraction labels", "Dual, auxiliary fraction"]
    colors = ["#546a7b", "#546a7b", "#829aab", "#147c76", "#28708f", "#80619a"]
    data = table.set_index("model").loc[list(NAMES)]
    ypos = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 4.3), sharey=True, layout="constrained")
    for ax, col, title in zip(axes, ("patient_ap", "event_f1", "false_per_h"),
                              ("Mean patient AP", "Event F1", "False events per hour")):
        vals = data[col].to_numpy()
        ax.barh(ypos, vals, color=colors)
        ax.set_xlim(0, max(vals)*1.31)
        ax.set_title(title)
        ax.spines[["right", "top"]].set_visible(False)
        for y, v in enumerate(vals):
            ax.text(v+max(vals)*.025, y, f"{v:.3f}" if col != "false_per_h" else f"{v:.1f}", va="center", fontsize=9)
    axes[0].set_yticks(ypos, titles)
    axes[0].invert_yaxis()
    fig.suptitle("Respiratory event detection on the same 30 held-out patients")
    fig.savefig(REPORT_DIR / "models.png", dpi=160)
    plt.close(fig)
