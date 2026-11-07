import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix,precision_recall_curve
from wearable_ppg.datasets.osasud import RESULT,RUN,LABELS,CACHE
from wearable_ppg.models.respiratory import PRIMARY,CONTROLS
from .metrics import metrics,event_rows,contiguous


def read_scores(folder,split):
    with np.load(folder/(split+".npz")) as z: return {k:z[k] for k in z.files}


def join(data):
    return {k:np.concatenate([a[k] for a in data]) for k in data[0]}


def evaluate(a):
    return metrics(a["y"],a["p"],a["subtype"],a["decision"],a["t"],a["patient"],a["arrival"])


def choose(table):
    eligible=table[table.val_ap>=table.val_ap.max()-.005]
    return eligible.sort_values(["val_event_f1","parameters","model"],ascending=[False,True,True]).iloc[0].model


def summarize(name,data):
    result,patients,pred=evaluate(data)
    for row in patients:
        ix=data["patient"]==row["patient"]; y=data["y"][ix]; p=data["p"][ix]
        row["ap"]=average_precision_score(y>0,p) if len(np.unique(y>0))==2 else np.nan
        row["positive_seconds"]=int((y>0).sum()); row["prevalence"]=float((y>0).mean()); row["model"]=name
    rng=np.random.default_rng(831)
    ap=np.asarray([r["ap"] for r in patients]); draws=rng.choice(ap,(3000,len(ap)),replace=True)
    result["ap_ci_lo"],result["ap_ci_hi"]=np.quantile(np.nanmean(draws,axis=1),[.025,.975])
    pr,re,f1,support=precision_recall_fscore_support(data["y"],pred,labels=np.arange(5),zero_division=0)
    classes=[dict(model=name,label=label,support=int(support[i]),patients=int(len(np.unique(data["patient"][data["y"]==i]))),
        precision=pr[i],recall=re[i],f1=f1[i]) for i,label in enumerate(LABELS)]
    return dict(model=name,**result),patients,classes


def paired_average_precision(patient):
    table = patient.pivot(index="patient", columns="model", values="ap")
    comparisons = [
        ("quality_cross_attention", "morphology_only"),
        ("quality_cross_attention", "modulation_only"),
        ("quality_cross_attention", "fixed_fusion"),
        ("cnn_transformer", "tcn_gru"),
    ]
    results = []
    for left, right in comparisons:
        difference = (table[left] - table[right]).dropna().to_numpy()
        draws = np.random.default_rng(831).choice(difference, (3000, len(difference)), replace=True)
        lower, upper = np.quantile(draws.mean(axis=1), [.025, .975])
        results.append(dict(candidate=left, reference=right, patients=len(difference),
                            mean_ap_difference=difference.mean(), lower_95=lower, upper_95=upper))
    return pd.DataFrame(results)


def compare():
    all_results=[]; by_patient=[]; by_class=[]; folds=[]; held={}; fold_choices=[]
    for name in PRIMARY+CONTROLS:
        parts=[]
        for k in range(5):
            path=RUN/name/f"fold{k}"; meta=json.loads((path/"done.json").read_text())
            test=read_scores(path,"test"); val=read_scores(path,"validation")
            assert not set(meta["test"])&set(meta["train"]+meta["val"]) and not set(meta["train"])&set(meta["val"])
            assert set(test["patient"])==set(meta["test"]) and set(val["patient"])==set(meta["val"])
            vrows,_=event_rows(val["y"],val["decision"],val["t"],val["patient"],val["arrival"])
            val_f1=np.mean([r["event_f1"] for r in vrows if r["true_events"]>0])
            measures,_,_=evaluate(test)
            folds.append(dict(model=name,fold=k,val_ap=meta["val_patient_ap"],val_event_f1=val_f1,
                parameters=meta["parameters"],epoch=meta["epoch"],threshold=meta["threshold"],**measures))
            parts.append(test)
        held[name]=parts
        r,patients,classes=summarize(name,join(parts)); all_results.append(r); by_patient+=patients; by_class+=classes
    fold=pd.DataFrame(folds); fold.to_csv(RESULT/"fold_metrics.csv",index=False)
    selection=fold[fold.model.isin(PRIMARY)]
    mean=selection.groupby("model",as_index=False)[["val_ap","val_event_f1","parameters"]].mean()
    winner=choose(mean); nested=[]
    for k in range(5):
        chosen=choose(selection[selection.fold==k]); fold_choices.append(chosen); nested.append(held[chosen][k])
    nested=join(nested)
    r,patients,classes=summarize("nested_selection",nested); all_results.append(r); by_patient+=patients; by_class+=classes
    ranking=pd.DataFrame(all_results).merge(fold.groupby("model",as_index=False)[["val_ap","val_event_f1","parameters"]].mean(),on="model",how="left")
    ranking.to_csv(RESULT/"comparison.csv",index=False)
    pd.DataFrame(by_patient).to_csv(RESULT/"patient_metrics.csv",index=False); pd.DataFrame(by_class).to_csv(RESULT/"class_metrics.csv",index=False)
    paired_average_precision(pd.DataFrame(by_patient)).to_csv(RESULT/"paired_ap.csv", index=False)
    selected=join(held[winner]); counts=np.unique(selected["patient"],return_counts=True)
    assert len(counts[0])==30 and len(selected["t"])==pd.read_csv(RESULT/"folds.csv").scored_seconds.sum()
    ref_events=covered_events=fully_covered_events=0
    for pid in counts[0]:
        with np.load(CACHE/"views"/f"p{pid:02}.npz") as z:
            t,y,ok=z["t"],z["y"],z["scorable"]
            ref=contiguous(y>0,t); ref_events+=len(ref)
            for begin,end in ref:
                i,j=np.searchsorted(t,[begin,end]); covered_events+=int(ok[i:j].any()); fully_covered_events+=int(ok[i:j].all())
    info=dict(winner=winner,outer_choices=fold_choices,reference_binary_runs=ref_events,runs_with_any_score=covered_events,
        runs_scored_completely=fully_covered_events,selection="mean inner patient AP; within 0.005 prefer validation patient event F1 then fewer parameters",
        nested=r)
    (RESULT/"selection.json").write_text(json.dumps(info,indent=2))
    figures(held,winner,ranking,pd.DataFrame(by_patient),selected)
    print(winner,ranking[["model","patient_ap","event_f1","five_macro_f1"]].round(3).to_string(index=False))


def figures(held,winner,ranking,patient,selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    path=RESULT/"figures"; path.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout(); fig.savefig(path/name,dpi=150,bbox_inches="tight"); plt.close(fig)
    main=ranking[ranking.model.isin(PRIMARY)]
    fig,ax=plt.subplots(figsize=(8,4))
    main.set_index("model")[["patient_ap","event_f1","five_macro_f1"]].plot.barh(ax=ax)
    ax.set(xlim=(0,1),xlabel="Held-out score",ylabel=""); save(fig,"models.png")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in PRIMARY:
        d=join(held[name]); y=d["y"]>0
        p,r,_=precision_recall_curve(y,d["p"]); axes[0].plot(r[::50],p[::50],label=name)
        real,prob=calibration_curve(y,d["p"],n_bins=12,strategy="quantile"); axes[1].plot(prob,real,".-")
    axes[0].axhline(y.mean(),ls="--",color="grey"); axes[0].legend(fontsize=7); axes[0].set(xlabel="Recall",ylabel="Precision")
    axes[1].plot([0,1],[0,1],"k--"); axes[1].set(xlabel="Estimated event probability",ylabel="Observed event fraction"); save(fig,"pr_calibration.png")
    fig,ax=plt.subplots(figsize=(6,5)); pred=np.where(selected["decision"],1+selected["subtype"].argmax(1),0)
    cm=confusion_matrix(selected["y"],pred,labels=np.arange(5),normalize="true")
    ax.imshow(cm,vmin=0,vmax=1,cmap="Blues"); labs=["None","Hypopnea","Central","Obstructive","Mixed"]
    ax.set(xticks=range(5),yticks=range(5),xticklabels=labs,yticklabels=labs,xlabel="Predicted",ylabel="Physician label")
    ax.tick_params(axis="x",rotation=30)
    for i in range(5):
        for j in range(5): ax.text(j,i,f"{cm[i,j]*100:.1f}%",ha="center",va="center",color="white" if cm[i,j]>.5 else "black")
    save(fig,"subtypes.png")
    fig,ax=plt.subplots(figsize=(9,4)); z=patient[patient.model==winner].sort_values("patient")
    ax.scatter(z.patient,z.ap,label="AP"); ax.scatter(z.patient,z.prevalence,label="Event prevalence",marker="x"); ax.legend()
    ax.set(xlabel="Held-out patient",ylabel="Fraction",ylim=(0,1)); save(fig,"patient_performance.png")
    fig,ax=plt.subplots(figsize=(8,4)); ctrl=ranking[ranking.model.isin(CONTROLS+("quality_cross_attention",))]
    ax.barh(ctrl.model,ctrl.patient_ap,color="#087f8c"); ax.set(xlabel="Mean patient AP",xlim=(0,max(.6,ctrl.patient_ap.max()+.03))); save(fig,"ablations.png")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in PRIMARY:
        histories=[json.loads((RUN/name/f"fold{k}"/"done.json").read_text())["history"] for k in range(5)]
        for i,metric in enumerate(("loss","patient_ap")):
            h=pd.DataFrame([{v["epoch"]:v[metric] for v in fold} for fold in histories]); axes[i].plot(h.columns,h.mean(),".-",label=name)
    for ax,lab in zip(axes,("Training loss","Validation patient AP")): ax.set(xlabel="Epoch (active folds)",ylabel=lab)
    axes[0].legend(fontsize=7); save(fig,"training.png")
