"""Apply each held-out binary model to that person's amusement and meditation."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE/"src"))
from wearable_eda.datasets.wesad import build_windows
from wearable_eda.evaluation.loso import SUBJECTS
from sklearn.metrics import balanced_accuracy_score, f1_score


def clock_control(cache, out):
    """Estimate how well session timing alone separates the lab blocks."""
    data = build_windows(cache,modalities=("eda",),calibration="zero")
    mid = (data["start"]+data["end"])/2
    groups, y = data["groups"], data["y"]
    rows = []
    for subject in SUBJECTS:
        train = groups!=subject
        thresholds = np.unique(mid[train])
        # Balance both classes inside each participant, then weight people equally.
        w = np.zeros(len(y))
        for s in SUBJECTS:
            if s!=subject:
                for label in (0,1):
                    use = (groups==s)&(y==label)
                    w[use] = 1/(28*use.sum())
        correct = (mid[:,None]>=thresholds)==y[:,None]
        threshold = float(thresholds[np.argmax(w[train] @ correct[train])])
        pred = mid[~train]>=threshold
        rows.append(dict(subject=subject,threshold_seconds=threshold,windows=len(pred),
                    macro_f1=f1_score(y[~train],pred,average="macro"),
                    balanced_accuracy=balanced_accuracy_score(y[~train],pred)))
    pd.DataFrame(rows).to_csv(out/"clock_control.csv",index=False)


def challenge(runs, cache, out):
    choices = pd.read_csv(out/"selection.csv",dtype={"subject":str})
    choices = choices[(choices.protocol=="zero") & (choices.rule=="lightweight") & (choices.subject!="refit")]
    if sorted(choices.subject.astype(int))!=list(SUBJECTS):
        raise ValueError("Expected one nested lightweight selection per test participant")
    views, rows = {}, []
    for row in choices.itertuples():
        subject = int(row.subject)
        family = row.selected.split("/")[0]
        folder = runs/row.selected
        spec = json.loads((folder/"meta.json").read_text())
        options = dict(modalities=spec["modalities"],calibration=spec["calibration"],
                       decomposition=spec["decomposition"],quality=spec.get("quality","auto"),
                       filtering=spec.get("filtering","conservative"))
        key = json.dumps(options,sort_keys=True)
        if key not in views:
            views[key] = build_windows(cache,classes=(1,2,3,4),**options)
        data = views[key]
        keep = (data["groups"]==subject) & (data["y"]>=2)
        if family=="classical":
            from wearable_eda.training.classical import load_bundle, predict_windows
            fitted = load_bundle(folder/"subjects"/f"S{subject}"/"model.joblib")
            if subject in fitted["train_subjects"]:
                raise ValueError("Held-out person appears in training")
            prob = predict_windows(fitted,data["F"][keep],data["features"])
        else:
            from wearable_eda.inference.neural import NeuralPredictor
            fitted = NeuralPredictor(folder/f"S{subject:02d}"/"model.pt")
            prob = fitted.predict_windows(data["X"][keep],data["F"][keep],data["channels"],data["features"])
        labels = data["y"][keep]
        for target, name in ((2,"amusement"),(3,"meditation")):
            values = prob[labels==target,1]
            if not len(values):
                raise ValueError(f"No {name} windows for S{subject}")
            rows.append(dict(subject=subject,experiment=row.selected,condition=name,windows=len(values),
                        stress_calls=int((values>.5).sum()),stress_call_fraction=float((values>.5).mean()),
                        mean_stress_score=float(values.mean())))
    detail = pd.DataFrame(rows)
    detail.to_csv(out/"protocol_challenge_subjects.csv",index=False)
    summary = detail.groupby("condition").agg(subjects=("subject","nunique"),windows=("windows","sum"),
              stress_calls=("stress_calls","sum"),mean_subject_call_fraction=("stress_call_fraction","mean"),
              between_subject_sd=("stress_call_fraction","std"),mean_subject_score=("mean_stress_score","mean"))
    summary.to_csv(out/"protocol_challenge.csv")
    fig,ax = plt.subplots(figsize=(7,3.6),layout="constrained")
    paired = detail.pivot(index="subject",columns="condition",values="stress_call_fraction")
    for _, values in paired.iterrows():
        ax.plot([0,1],values[["amusement","meditation"]],"-o",color="#90aaa8",linewidth=.8,ms=3)
    ax.plot([0,1],paired[["amusement","meditation"]].mean(),"-o",color="#0a504b",linewidth=2.4,label="Mean across participants")
    ax.set(xticks=[0,1],xticklabels=["Amusement","Meditation"],xlim=(-.35,1.35),ylim=(-.03,1.03),
           ylabel="Fraction of windows called stress",title="Binary model on two held-out protocol states")
    ax.legend(fontsize=8)
    fig.savefig(out/"protocol_challenge.png",dpi=150)
    plt.close(fig)
    print(summary.to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("artifacts/eda/runs/wesad_loso_v1"))
    p.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    p.add_argument("--out",type=Path,default=Path("research/results/wesad_loso_v1"))
    args = p.parse_args()
    challenge(args.runs,args.cache,args.out)
    clock_control(args.cache,args.out)
