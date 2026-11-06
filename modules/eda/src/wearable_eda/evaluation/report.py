"""Create portable figures and metric tables from authenticated LOSO records."""
import json
from hashlib import sha256
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from .loso import SUBJECTS, load_runs, paired, require_complete, select, summarize, verify_annotations


def publish_audits(root, out, runs):
    folder = out/"verification"
    folder.mkdir(exist_ok=True)
    entries = []
    for run in runs.values():
        if run.family=="nested":
            continue
        row = {"experiment":run.name,"participants":len(run.folds),"windows":len(run.pred)}
        for file in ("meta.json","folds.csv","predictions.csv"):
            row[file+"_sha256"] = sha256((root/run.name/file).read_bytes()).hexdigest()
        entries.append(row)
    pd.DataFrame(entries).to_csv(folder/"run_index.csv",index=False)
    evidence = {"classical/cpu_benchmark.csv":"classical_host.csv",
                "neural_audit/host_inference.csv":"neural_host.csv",
                "neural_audit/continuous_decomposition.csv":"learned_components.csv",
                "neural_audit/checkpoint_replay.csv":"neural_checkpoint_replay.csv",
                "neural_audit/learning_curves.png":"learning_curves.png",
                "neural_audit/held_out_decomposition.png":"learned_decomposition.png"}
    for source, dest in evidence.items():
        if (root/source).is_file():
            shutil.copyfile(root/source,folder/dest)


def write_comparison(summary, out):
    lines = ["# Complete WESAD model comparison", "",
             "Each row summarizes the same 15 held-out participants with equal participant weights. Macro F1 includes the between-participant sample SD. The binary sensitivity and specificity columns refer to the stress class. Multiclass columns use macro one-vs-rest averaging. Size is the median fold checkpoint size in KiB. Parameter counts apply to linear and neural models.", "",
             "See the [interpretation](README.md), [all metrics and confidence intervals](all_models.csv) and [per-participant results](per_subject.csv).", ""]
    for title,frame in (("Zero-calibration primary candidates",summary[summary.role.isin(["primary","candidate"]) & (summary.calibration=="zero")]),
                        ("Resting-calibrated primary candidates",summary[summary.role.isin(["primary","candidate"]) & (summary.calibration=="rest")]),
                        ("Ablations and budget sensitivities",summary[summary.role.isin(["ablation","budget_sensitivity"])]),
                        ("Context-only controls",summary[summary.role=="negative_control"]),
                        ("Three- and four-condition experiments",summary[summary.role=="secondary"]),
                        ("Selection repeated within each outer fold",summary[summary.role=="nested"])):
        if frame.empty:
            continue
        lines.extend(["## "+title, "", "| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in frame.sort_values(["classes","category","experiment"]).itertuples():
            params = f"{row.trainable_parameters:,.0f}" if pd.notna(row.trainable_parameters) else "—"
            size = f"{row.artifact_bytes/1024:.1f}" if pd.notna(row.artifact_bytes) else "—"
            lines.append(f"| `{row.experiment}` | {row.modalities} | {row.accuracy:.3f} | {row.macro_f1:.3f} ± {row.macro_f1_sd:.3f} | {row.balanced_accuracy:.3f} | {row.sensitivity:.3f} | {row.specificity:.3f} | {row.auroc:.3f} | {row.auprc:.3f} | {row.brier:.3f} | {params} | {size} |")
        lines.append("")
    (out/"model_comparison.md").write_text("\n".join(lines),encoding="utf-8")


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def category_leaders(summary, choices):
    rows = []
    for protocol in ("zero","rest"):
        candidates = summary[(summary.calibration==protocol) & (summary.classes==2)
                        & summary.role.isin(["primary","candidate"])]
        if candidates.empty:
            continue
        groups = {"Traditional":candidates.category=="traditional", "Standalone DL":candidates.category=="standalone",
                  "Hybrid DL":candidates.category=="hybrid","Self-supervised":candidates.category=="ssl",
                  "Any DL":candidates.family=="neural", "EDA only":candidates.modalities=="eda",
                  "EDA + motion":candidates.modalities=="eda+acc", "All candidates":np.ones(len(candidates),bool)}
        for label, mask in groups.items():
            subset = candidates[mask]
            if subset.empty:
                continue
            inner = subset.sort_values(["inner_macro_f1","experiment"], ascending=[False,True]).iloc[0]
            observed = subset.sort_values(["macro_f1","experiment"], ascending=[False,True]).iloc[0]
            rows.append(dict(group=label,protocol=protocol,inner_choice=inner.experiment,
                    inner_macro_f1=inner.inner_macro_f1,choice_outer_macro_f1=inner.macro_f1,
                    descriptive_highest=observed.experiment,highest_outer_macro_f1=observed.macro_f1))
        for c in choices:
            if c["protocol"]==protocol and c["rule"]=="lightweight" and c["subject"]=="refit":
                row = candidates[candidates.experiment==c["selected"]].iloc[0]
                rows.append(dict(group="Lightweight",protocol=protocol,inner_choice=row.experiment,
                          inner_macro_f1=row.inner_macro_f1,choice_outer_macro_f1=row.macro_f1))
    return pd.DataFrame(rows)


def fixed_model_charts(out, leaders, runs):
    zero = leaders[leaders.protocol=="zero"].set_index("group")
    if "Any DL" not in zero.index or "Lightweight" not in zero.index:
        return
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    for i,group in enumerate(("Lightweight","Any DL")):
        run = runs[zero.loc[group,"inner_choice"]]
        p = run.pred
        fpr,tpr,_ = metrics.roc_curve(p.y_true,p.p1)
        precision,recall,_ = metrics.precision_recall_curve(p.y_true,p.p1)
        axes[i,0].plot(fpr,tpr,color="#0a695b")
        axes[i,0].plot([0,1],[0,1],"--",color="grey")
        axes[i,0].set(xlabel="False-positive rate", ylabel="Stress sensitivity",title="ROC")
        axes[i,1].step(recall,precision,where="post",color="#276e91")
        axes[i,1].axhline(p.y_true.mean(),ls="--",color="grey")
        axes[i,1].set(xlabel="Stress recall", ylabel="Stress precision",title="Precision–recall")
        for ax in axes[i,:2]:
            ax.set(xlim=(0,1),ylim=(0,1.02))
        cm = metrics.confusion_matrix(p.y_true,p[["p0","p1"]].to_numpy().argmax(1),labels=[0,1])
        metrics.ConfusionMatrixDisplay(cm,display_labels=["Baseline","Stress"]).plot(ax=axes[i,2],cmap="Blues",colorbar=False)
        axes[i,2].set_title(run.name.split("/")[1].replace("__"," · ").replace("_"," "))
    fig.suptitle("Fixed deployment candidates · held-out windows pooled across 15 participants")
    finish(fig,out/"export_candidates_curves.png")


def charts(out, summary, subjects, runs):
    prim = summary.query("role in ['primary','candidate'] and calibration=='zero' and classes==2")
    for sensors, filename, title in (("eda","eda_only_architectures","Wrist EDA only"),
                                      ("eda+acc","architectures","Wrist EDA and motion")):
        frame = prim[prim.modalities==sensors].sort_values("macro_f1")
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(4, .30*len(frame))))
        y = np.arange(len(frame))
        ax.hlines(y, frame.macro_f1_lo, frame.macro_f1_hi, color="#749198")
        ax.scatter(frame.macro_f1, y, color="#12675d", s=20)
        labels = [row.experiment.split("/")[-1].removesuffix("_zero") if row.family=="neural"
                  else row.model for row in frame.itertuples()]
        ax.set(yticks=y, yticklabels=[label.replace("_"," ") for label in labels], xlim=(0,1.01),
               xlabel="Mean participant macro F1 and 95% participant-bootstrap interval",
               title=title+" · zero calibration")
        ax.grid(axis="x", alpha=.25)
        finish(fig, out/(filename+".png"))
    for protocol in ("zero", "rest"):
        keys = [f"nested/{protocol}_{mode}" for mode in ("highest_inner","lightweight")]
        if keys[0] not in runs:
            continue
        fig, axes = plt.subplots(1,3,figsize=(12,3.7))
        fig.suptitle(f"Nested model selection · {protocol} calibration · pooled held-out windows")
        fig2, (cm_ax, sub_ax) = plt.subplots(1,2,figsize=(10,4))
        for key, label in zip(keys, ("Highest inner score", "Lightweight rule")):
            p = runs[key].pred
            fpr, tpr, _ = metrics.roc_curve(p.y_true, p.p1)
            precision, recall, _ = metrics.precision_recall_curve(p.y_true, p.p1)
            axes[0].plot(fpr,tpr,label=label)
            axes[1].step(recall,precision,where="post")
            bins = pd.cut(p.p1,np.linspace(0,1,11),include_lowest=True)
            rel = p.groupby(bins,observed=True).agg(score=("p1","mean"), observed=("y_true","mean"), n=("p1","size"))
            rel.to_csv(out / f"{protocol}_{key.split('_')[-1]}_reliability.csv")
            axes[2].plot(rel.score,rel.observed, marker="o", ms=4)
            sub = subjects[subjects.experiment == key]
            sub_ax.plot(sub.subject.astype(str),sub.macro_f1,marker="o",ms=4,label=label)
        axes[0].plot([0,1],[0,1],"--",color="grey")
        axes[1].axhline(p.y_true.mean(),ls="--",color="grey")
        axes[2].plot([0,1],[0,1],"--",color="grey")
        for ax, x, y in zip(axes,("False-positive rate","Stress recall","Predicted stress probability"),
                                ("Stress sensitivity","Stress precision","Observed stress fraction")):
            ax.set(xlabel=x,ylabel=y,xlim=(0,1),ylim=(0,1.025))
        axes[0].legend(fontsize=8, loc="lower right")
        finish(fig, out/f"{protocol}_roc_pr_calibration.png")
        d = runs[keys[1]].pred
        cm = metrics.confusion_matrix(d.y_true,d[["p0","p1"]].to_numpy().argmax(1),labels=[0,1])
        metrics.ConfusionMatrixDisplay(cm,display_labels=["Baseline","Stress"]).plot(ax=cm_ax,cmap="Blues",colorbar=False)
        cm_ax.set_title("Lightweight rule · pooled windows")
        sub_ax.set(xlabel="Held-out subject",ylabel="Macro F1",ylim=(0,1.025),title="Each participant has equal weight")
        sub_ax.legend(fontsize=8)
        finish(fig2, out/f"{protocol}_subjects_confusion.png")


def motion_analysis(cache, run, out):
    # Device-native 32 Hz acceleration. Remove the within-second mean vector and
    # express RMS in g. Thresholds for a test subject use other participants only.
    p = run.pred.copy()
    p["motion_rms_g"] = np.nan
    for subject, rows in p.groupby("subject"):
        with np.load(cache/f"S{subject}.npz") as bag:
            xyz = np.asarray(bag["acc"], dtype=np.float64).reshape(-1,3)/64
        whole = xyz[:len(xyz)//32*32].reshape(-1,32,3)
        energy = np.nanmean(np.nansum((whole - np.nanmean(whole,axis=1,keepdims=True))**2, axis=2),axis=1)
        for idx, start in rows.start.items():
            a = int(np.ceil(start))
            b = int(np.floor(start+60))
            p.at[idx,"motion_rms_g"] = np.sqrt(np.nanmean(energy[a:b]))
    out_rows = []
    for subject in SUBJECTS:
        other = p.loc[p.subject != subject,"motion_rms_g"]
        cuts = other.quantile([1/3,2/3]).to_numpy()
        held = p[p.subject == subject].copy()
        held["stratum"] = np.searchsorted(cuts, held.motion_rms_g)
        for group, block in held.groupby("stratum"):
            y, pred = block.y_true.to_numpy(), block[["p0","p1"]].to_numpy().argmax(1)
            pos, neg = y==1, y==0
            out_rows.append(dict(subject=subject,motion=("low","middle","high")[group],
                                 low_cut_g=cuts[0], high_cut_g=cuts[1], windows=len(y),
                                 positives=pos.sum(), negatives=neg.sum(), accuracy=(y==pred).mean(),
                                 sensitivity=pred[pos].mean() if pos.any() else np.nan,
                                 fpr=pred[neg].mean() if neg.any() else np.nan,
                                 brier=((y-block.p1.to_numpy())**2).mean()))
    detail = pd.DataFrame(out_rows)
    detail.to_csv(out/"motion_strata_subjects.csv",index=False)
    detail.groupby("motion").agg(subjects=("subject","nunique"), windows=("windows","sum"),
        positives=("positives","sum"), negatives=("negatives","sum"), sensitivity=("sensitivity","mean"),
        subjects_with_stress=("sensitivity","count"), fpr=("fpr","mean"), subjects_with_baseline=("fpr","count"),
        brier=("brier","mean")).to_csv(out/"motion_strata.csv")


def make_report(root, out, cache=None, contrasts=None, partial=False, exploratory=None):
    root, out = Path(root), Path(out)
    runs = load_runs(root)
    if not partial:
        require_complete(runs)
    if cache:
        verify_annotations(runs,cache)
    choices = []
    for protocol in ("zero","rest"):
        new, nested = select(runs,protocol)
        choices.extend(new)
        runs.update({r.name:r for r in nested})
    rows, people = [], []
    for run in runs.values():
        row, part = summarize(run)
        rows.append(row)
        people.extend(part)
    summary = pd.DataFrame(rows)
    sub = pd.DataFrame(people)
    out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"all_models.csv",index=False)
    write_comparison(summary,out)
    sub.to_csv(out/"per_subject.csv",index=False)
    pd.DataFrame(choices).to_csv(out/"selection.csv",index=False)
    leaders = category_leaders(summary, choices)
    leaders.to_csv(out/"category_leaders.csv",index=False)
    if not leaders.empty:
        fixed_model_charts(out,leaders,runs)
    if contrasts:
        pairs = json.loads(Path(contrasts).read_text())
        missing = {key for _,a,b in pairs for key in (a,b) if key not in runs}
        if missing:
            raise ValueError(f"Unfinished contrast experiments: {sorted(missing)}")
        effects = paired(pairs, sub, runs)
        effects.to_csv(out/"paired_contrasts.csv",index=False)
        fig,ax = plt.subplots(figsize=(9,5))
        primary_effects = effects.head(10).iloc[::-1]
        labels = [x.split(" (")[0] for x in primary_effects.contrast]
        ax.hlines(labels,primary_effects.low,primary_effects.high,color="#8da29d")
        ax.scatter(primary_effects.delta_macro_f1,labels,color="#12675d")
        ax.axvline(0, color="grey", ls="--")
        ax.set(xlabel="Mean paired difference in participant macro F1 (95% bootstrap interval)",
               title="Incremental contributions on the same 15 participants")
        finish(fig,out/"component_ablations.png")
    if exploratory:
        pairs = json.loads(Path(exploratory).read_text())
        subset = paired(pairs,sub,runs)
        if len(subset)!=len(pairs):
            raise ValueError("Missing an exploratory comparison")
        subset.to_csv(out/"exploratory_contrasts.csv",index=False)
    charts(out,summary,sub,runs)
    if cache and "nested/zero_lightweight" in runs:
        motion_analysis(Path(cache),runs["nested/zero_lightweight"],out)
    publish_audits(root,out,runs)
    counts = summary.groupby(["family","classes"]).size().to_dict()
    print(f"Verified {len(rows)} experiment/selection results. Counts {counts}. Output {out.resolve()}")
