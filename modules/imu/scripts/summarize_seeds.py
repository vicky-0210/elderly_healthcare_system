"""Secondary stability check on the original fixed participant folds."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from wearable_imu.evaluation.compare import metrics,decorate,fixed,bootstrap
from wearable_imu.evaluation.figures import plt,plotfile
from wearable_imu.training.protocol import RUN
from wearable_imu.datasets.sisfall import RESULTS,CACHE

if __name__=="__main__":
    totals=[]; trials=[]; paired=[]
    manifest=pd.read_csv(CACHE/"manifest.csv")
    for seed in (4907,6203,7919):
        root=RUN if seed==4907 else RUN.with_name(RUN.name+f"_seed{seed}")
        for key in ("dm_tcn","dm_standard","inception","dm_distilled"):
            parts=[]
            for k in range(5):
                folder=root/key/f"fold{k}"
                meta=json.loads((folder/"done.json").read_text())
                assert set(meta["held_subjects"]).isdisjoint(meta["refit_subjects"])
                parts.append(decorate(pd.read_csv(folder/"test.csv"),folder,meta,manifest))
            r=pd.concat(parts,ignore_index=True)
            if len(r)!=4500 or r.id.duplicated().any() or r.subject.nunique()!=38: raise AssertionError("Incomplete seed")
            ci=bootstrap(r)
            totals.append(dict(seed=seed,model=key,ci_low=ci[0],ci_high=ci[1],fixed8_subject_ba=metrics(fixed(r))["subject_ba"],**metrics(r)))
            trials.append(r.assign(seed=seed,model=key))
            if key=="dm_standard": control=r
            if key=="dm_distilled":
                lo,hi=bootstrap(r,control)
                paired.append(dict(seed=seed,delta=metrics(r)["subject_ba"]-metrics(control)["subject_ba"],low=lo,high=hi))
    pd.DataFrame(paired).to_csv(RESULTS/"distillation_paired.csv",index=False)
    stats=pd.DataFrame(totals); stats.to_csv(RESULTS/"replicate_seeds.csv",index=False)
    pd.concat(trials,ignore_index=True).to_csv(RESULTS/"replicate_trials.csv.gz",compression="gzip",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for key,rows in stats.groupby("model"): ax.plot(rows.seed.astype(str),rows.subject_ba,"o-",label=key)
    ax.legend(); ax.set(xlabel="Training seed; same five participant groups",ylabel="Subject-balanced accuracy",title="Sensitivity to initialization and minibatch order")
    plotfile(fig,RESULTS/"figures"/"seeds.png"); print(stats[["seed","model","subject_ba","recall","specificity"]].to_string(index=False))
