"""Add the predeclared paired comparisons to the shareable result and verify local references."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from study import A,C,R,save_json


def main():
    dev=pd.read_csv(R/"development.csv").set_index("name")
    sel=json.loads((R/"selection.json").read_text())
    name=sel["student"]; teacher=sel["teacher"]
    tab=pd.read_csv(R/"final.csv");tab=tab[(tab.unit=="patient")&(tab.target_spec==.90)].set_index("name")
    kd=pd.read_csv(R/"paired_kd.csv")
    joint=pd.read_csv(R/"joint_ablation.csv").set_index("name")
    coverage=pd.read_csv(R/"confidence_coverage.csv")
    w=json.loads((R/"wearable_window_metrics.json").read_text())
    e=json.loads((R/"export.json").read_text())
    groups=pd.read_csv(R/"signal_integrity.csv").sum()
    configs={p.parent.name:json.loads(p.read_text())["config"] for p in (A/"development_amp").glob("*/done.json")}
    save_json(R/"configurations.json",configs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.calibration import calibration_curve
    from study import patient
    from scipy.special import expit
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    meta=pd.read_csv(C/"manifest.csv")
    for nm,label in ((teacher,"12-lead teacher"),(name,"Selected Lead II"),("joint_full","Full distilled Lead II")):
        if nm=="E0":continue
        with np.load(A/"refit"/nm/"scores.npz") as q:
            frame=meta.iloc[q["ids"]];hold=frame.strat_fold==10
            pts=patient(frame[hold],q["logit"][hold])
        ab=json.loads((A/"calibration"/(nm+".json")).read_text())
        p=expit(pts.z*ab["slope"]+ab["intercept"])
        f,t,_=roc_curve(pts.y,p);axes[0].plot(f,t,label=label)
        pr,rc,_=precision_recall_curve(pts.y,p);axes[1].plot(rc,pr)
        obs,prob=calibration_curve(pts.y,p,n_bins=10,strategy="quantile");axes[2].plot(prob,obs,".-")
    axes[0].plot([0,1],[0,1],"k--",lw=.8);axes[2].plot([0,1],[0,1],"k--",lw=.8)
    axes[1].axhline(pts.y.mean(),ls="--",color="gray")
    for ax,labs in zip(axes,[("False-positive rate","Sensitivity"),("Recall","Precision"),("Predicted probability","Observed fraction")]):
        ax.set(xlabel=labs[0],ylabel=labs[1],xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8);fig.savefig(R/"heldout_curves.png",dpi=160);plt.close(fig)
    lines=["## Distillation and component evidence","",
        f"The selected student ({name}) achieved patient AP {tab.loc[name,'auprc']:.3f}, compared with {tab.loc['E0','auprc']:.3f} for the reproduced CNN and {tab.loc[teacher,'auprc']:.3f} for the 12-lead teacher. All three use the same eligible patients. The teacher–student difference measures this architecture and cohort; it is not an information-theoretic limit.",
        "", "| Patient AP contrast on fold 10 | Difference | Paired 95% interval |","|---|---:|---:|"]
    for r in kd.itertuples():
        lines.append(f"| {r.model} minus {r.reference} | {r.ap_difference:+.3f} | [{r.low:+.3f}, {r.high:+.3f}] |")
    lines.extend(["","Intervals use 1,000 paired patient bootstraps with frozen checkpoints. Each challenger was trained with one seed. Sampling intervals exclude training-seed variability. [KD final metrics at four operating points](kd_final.csv) · [Paired contrasts](paired_kd.csv)",
        "",f"The full joint morphology and teaching model reached development AP {joint.loc['joint_full','ap']:.3f}. Removing all teacher supervision gave {joint.loc['joint_no_teacher','ap']:.3f}; removing the explicit morphology fusion path gave {joint.loc['joint_no_morphology','ap']:.3f}. Interpret these as interactions within this loss and backbone. [Joint component removals](joint_ablation.csv) · [Effective model configurations](configurations.json)",
        "",f"Confidence filtering at likelihood 100 retained {coverage.iloc[-1].coverage:.1%} of development records and {int(coverage.iloc[-1].positives)} positive records. On the full source-window audit {int(groups.nonfinite)} recordings contained nonfinite samples, {int(groups.whole_flat)} were entirely flat, {int(groups.over_5mv)} exceeded ±5 mV, and {int(groups.plateau_100ms)} had a constant stretch of at least 50 samples. A constant stretch can occur at the stored record boundary. This flag alone cannot establish ADC clipping. [Signal audit](signal_integrity.csv) · [Patient split audit](split_audit.json)",
        "",f"The wearable ten-second score uses separate fold-9 record-level calibration. At its frozen threshold, fold-10 window sensitivity was {w['sensitivity']:.3f}, specificity {w['specificity']:.3f}, and Brier {w['brier']:.3f}. The reported raw-input parity checked {e['checked']} source records. [Window metrics](wearable_window_metrics.json)",
        "", "The mixed-precision challenger runs are under development_amp; earlier full-precision timing pilots are excluded. Ranking uses one fixed budget across challengers. The greedy E10 search tests selected combinations and leaves untested interactions. Synthetic perturbations and retrospective clinical waveforms leave wearable electrode placement, gain, artifact and population shifts unmeasured."])
    lines.extend(["","[Held-out ROC, precision–recall and calibration plots](heldout_curves.png) · [BioAmp board reference](https://docs.upsidedownlabs.tech/hardware/bioamp/bioamp-exg-pill/index.html)"])
    p=R/"REPORT.md";s=p.read_text(encoding="utf-8").split("## Distillation and component evidence")[0].rstrip()
    p.write_text(s+"\n\n"+"\n".join(lines)+"\n",encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8")):
        if not link.startswith("http"): assert (p.parent/link).exists(),link
    print(tab[["auprc","auroc","sensitivity","specificity","f1"]].round(4));print(kd.round(4).to_string(index=False))

if __name__=="__main__":main()
