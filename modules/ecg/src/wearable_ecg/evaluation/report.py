"""Final frozen comparison and reproducible PNG/PDF plots."""
from __future__ import annotations
import json
import time
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from wearable_ecg.datasets.prepare import load_cache
from wearable_ecg.labels.ptbxl import cohort
from wearable_ecg.models.backbones import ECGClassifier
from wearable_ecg.training.runner import tensors,predict_logits
from wearable_ecg.evaluation.metrics import (metrics,bootstrap,paired_bootstrap,fit_calibration,
                                            calibrated,choose_threshold)
from wearable_ecg.runtime import save_json,digest,seed_everything

COLORS={"cnn":"#1976a3","cnn_transformer":"#c44c30","cnn_ssm":"#618b32",
        "voltage_logistic":"#8d5ab5","age_sex_logistic":"#6c7179","quality_logistic":"#ad8c30",
        "voltage_random_forest":"#20968b"}
DISPLAY={"cnn":"CNN","cnn_transformer":"CNN + Transformer","cnn_ssm":"CNN + diagonal SSM",
         "voltage_logistic":"Voltage/morphology logistic","age_sex_logistic":"Age/sex control",
         "quality_logistic":"Quality control","voltage_random_forest":"Voltage/morphology RF"}


def save_figure(fig,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def subgroup_metrics(test,p,threshold):
    rows=[]
    selectors={"all":np.ones(len(test),bool),"age_65_plus":test.age65plus.astype(bool).to_numpy(),
               "age_under_65":(test.age<65).to_numpy(),"sex_0":(test.sex==0).to_numpy(),
               "sex_1":(test.sex==1).to_numpy(),"high_confidence_lvh_vs_negatives":((test.label==0)|(test.lvh_confidence==100)).to_numpy(),
               "isolated_lvh_vs_negatives":((test.label==0)|test.isolated_lvh.astype(bool)).to_numpy(),
               "comorbid_lvh_vs_negatives":((test.label==0)|(~test.isolated_lvh.astype(bool))).to_numpy(),
               "stt_comorbidity":test.stt_comorbidity.astype(bool).to_numpy(),
               "conduction_comorbidity":test.conduction_comorbidity.astype(bool).to_numpy()}
    for name,mask in selectors.items():
        if mask.sum(): rows.append({"subgroup":name,**metrics(test.label.to_numpy()[mask],p[mask],threshold)})
    return rows


def evaluate_run(cache,run,n_boot=1000):
    run=Path(run); out=run/"test"
    if (out/"metrics.json").exists(): return json.loads((out/"metrics.json").read_text())
    cfg=json.loads((run/"config.json").read_text()); cal=json.loads((run/"calibration.json").read_text())
    done=json.loads((run/"COMPLETE.json").read_text())
    if digest(run/"refit/model.pt")!=done["model_sha256"]: raise ValueError("Checkpoint changed after freeze")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])
    frame,waves,features=load_cache(cache,cfg["preprocessing"])
    test=cohort(frame,cfg["task"]).query("strat_fold == 10")
    checkpoint=torch.load(run/"refit/model.pt",map_location="cpu",weights_only=True)
    model=ECGClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(device)
    x,f,y=tensors(waves,features,test,device)
    logits=predict_logits(model,x,f); p=calibrated(logits,cal)
    out.mkdir(parents=True,exist_ok=True)
    columns=["ecg_id","patient_id","strat_fold","label","age","sex","lvh_confidence","isolated_lvh","stt_comorbidity","conduction_comorbidity"]
    predictions=test[columns].copy(); predictions["logit"]=logits; predictions["probability"]=p
    predictions["prediction"]=p>=cal["threshold"]; predictions["quality"]=np.asarray(features[test.row_index,14])
    predictions.to_csv(out/"predictions.csv",index=False)
    result=metrics(test.label,p,cal["threshold"])
    result.update({"task":cfg["task"],"architecture":cfg["architecture"],"seed":cfg["seed"],
                   "variant":cfg["preprocessing"],"parameters":done["refit"]["parameters"],
                   "selected_epochs":done["development"]["best_epoch"],
                   "training_seconds":done["refit"]["training_seconds"]+done["development"]["training_seconds"]})
    save_json(out/"patient_bootstrap_ci.json",bootstrap(test.label.to_numpy(),p,test.patient_id,cal["threshold"],n_boot=n_boot))
    save_json(out/"high_sensitivity_operating_point.json",metrics(test.label,p,cal["high_sensitivity_threshold"]))
    pd.DataFrame(subgroup_metrics(test,p,cal["threshold"])).to_csv(out/"subgroups.csv",index=False)
    # Desktop-only batched-window compute, explicitly excluding preparation and transfer.
    model.eval(); duration=[]
    with torch.inference_mode():
        for i in range(70):
            if device.type=="cuda": torch.cuda.synchronize()
            t=time.perf_counter()
            model(x[:1],f[:1])
            if device.type=="cuda": torch.cuda.synchronize()
            if i>=10: duration.append((time.perf_counter()-t)*1000)
    save_json(out/"desktop_latency.json",{"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        "precision":"FP32","batch_size":1,"warmup":10,"repetitions":60,
        "p50_ms":float(np.median(duration)),"p95_ms":float(np.quantile(duration,.95)),
        "excludes":"preprocessing, acquisition, host-device transfer; not Jetson Nano timing"})
    save_json(out/"metrics.json",result)
    del model,x,f,y; torch.cuda.empty_cache()
    return result


def classical_controls(cache,root,task):
    frame,waves,features=load_cache(cache)
    selected=cohort(frame,task)
    train=selected.query("strat_fold <= 8"); cal=selected.query("strat_fold == 9"); test=selected.query("strat_fold == 10")
    rows=[]
    for name in ("voltage_logistic","voltage_random_forest","age_sex_logistic","quality_logistic"):
        out=Path(root)/task/name; out.mkdir(parents=True,exist_ok=True)
        if (out/"metrics.json").exists(): rows.append(json.loads((out/"metrics.json").read_text())); continue
        if name=="age_sex_logistic":
            allx=frame[["age_for_control","age_encoded","sex"]].astype(float).to_numpy()
        elif name=="quality_logistic":
            # Acquisition quality/delineation indicators, no amplitude/age/report.
            allx=np.asarray(features[:,12:16])
        else: allx=np.asarray(features[:,:12])
        estimator=(RandomForestClassifier(n_estimators=300,min_samples_leaf=8,max_features="sqrt",class_weight="balanced",n_jobs=2,random_state=42)
                   if name=="voltage_random_forest" else LogisticRegression(max_iter=2000,class_weight="balanced",C=1.,random_state=42))
        model=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),estimator)
        model.fit(allx[train.row_index],train.label)
        def scores(index):
            p=np.clip(model.predict_proba(allx[index])[:,1],1e-6,1-1e-6)
            return np.log(p/(1-p))
        logits=scores(cal.row_index); calibrator=fit_calibration(logits,cal.label)
        cp=calibrated(logits,calibrator); threshold=choose_threshold(cal.label,cp)
        save_json(out/"calibration.json",{**calibrator,"threshold":threshold})
        pd.DataFrame({"ecg_id":cal.ecg_id,"label":cal.label,"probability":cp}).to_csv(out/"calibration_predictions.csv",index=False)
        joblib.dump(model,out/"model.joblib")
        p=calibrated(scores(test.row_index),calibrator)
        predicted=test[["ecg_id","patient_id","label","age","sex"]].copy(); predicted["probability"]=p
        predicted.to_csv(out/"predictions.csv",index=False)
        result={"task":task,"architecture":name,"seed":42,"variant":"constrained",**metrics(test.label,p,threshold)}
        save_json(out/"metrics.json",result); rows.append(result)
    return rows


def plot_task(root,task,comparison):
    root=Path(root); plots=root/"plots"/task; taskroot=root/task
    models={}; cal_probs={}; thresholds={}; predictions_by_seed={}
    for arch in ("cnn","cnn_transformer","cnn_ssm"):
        runs=sorted(taskroot.glob(f"{arch}__constrained__seed*"))
        if not runs: continue
        preds=[pd.read_csv(r/"test/predictions.csv").sort_values("ecg_id") for r in runs]
        cals=[pd.read_csv(r/"calibration_predictions.csv").sort_values("ecg_id") for r in runs]
        if not all(np.array_equal(preds[0].ecg_id,p.ecg_id) for p in preds): raise ValueError("Unpaired predictions")
        df=preds[0].copy(); df.probability=np.mean([p.probability.to_numpy() for p in preds],axis=0)
        models[arch]=df; predictions_by_seed[arch]=preds
        cp=np.mean([p.probability.to_numpy() for p in cals],axis=0)
        thresholds[arch]=choose_threshold(cals[0].label,cp)
        cal_probs[arch]=cp
    for name in ("voltage_logistic","age_sex_logistic","quality_logistic","voltage_random_forest"):
        models[name]=pd.read_csv(taskroot/name/"predictions.csv").sort_values("ecg_id")
        thresholds[name]=json.loads((taskroot/name/"calibration.json").read_text())["threshold"]
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for name,df in models.items():
        y=df.label.to_numpy(); p=df.probability.to_numpy(); m=metrics(y,p,thresholds[name])
        fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
        style="-" if name in predictions_by_seed else "--"
        axes[0].plot(fpr,tpr,style,color=COLORS[name],label=f"{DISPLAY[name]} ({m['auroc']:.3f})")
        axes[1].plot(recall,precision,style,color=COLORS[name],label=DISPLAY[name])
        if name in predictions_by_seed:
            observed,predicted=calibration_curve(y,p,n_bins=10,strategy="quantile")
            axes[2].plot(predicted,observed,"o-",color=COLORS[name],label=DISPLAY[name])
    axes[0].plot([0,1],[0,1],":",color="gray"); axes[0].set(xlabel="False-positive rate",ylabel="Sensitivity",title="ROC · frozen test fold 10")
    axes[1].axhline(y.mean(),color="gray",linestyle=":"); axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall")
    axes[2].plot([0,1],[0,1],":",color="gray"); axes[2].set(xlabel="Mean calibrated probability",ylabel="Observed LVH annotation rate",title="Reliability · equal-count bins")
    axes[0].legend(fontsize=7); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=.18)
    fig.suptitle(f"{task.replace('_',' ')} · Lead II · neural curves use a five-seed probability ensemble",fontsize=12)
    fig.tight_layout(); save_figure(fig,plots/"roc_pr_calibration")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    ensemble_results=[]
    for ax,(name,preds) in zip(axes,predictions_by_seed.items()):
        df=models[name]; cm=confusion_matrix(df.label,df.probability>=thresholds[name],labels=[0,1])
        ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()*.5 else "black",fontsize=16)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","LVH pattern"],yticklabels=["Negative","LVH pattern"],xlabel="Predicted",ylabel="Annotation",title=DISPLAY[name])
        m=metrics(df.label,df.probability,thresholds[name]); ensemble_results.append({"architecture":name,**m})
        save_json(taskroot/f"{name}_ensemble_patient_ci.json",bootstrap(df.label.to_numpy(),df.probability.to_numpy(),df.patient_id,thresholds[name],1000))
        df.to_csv(taskroot/f"{name}_ensemble_predictions.csv",index=False)
    fig.suptitle("Five-seed ensembles · thresholds selected on fold 9"); fig.tight_layout(); save_figure(fig,plots/"confusion_matrices")
    pd.DataFrame(ensemble_results).to_csv(taskroot/"ensemble_metrics.csv",index=False)
    pairs=[]
    for a,b in itertools.combinations(predictions_by_seed,2):
        da,db=models[a],models[b]
        pairs.append({"model_a":a,"model_b":b,**paired_bootstrap(da.label.to_numpy(),da.probability.to_numpy(),db.probability.to_numpy(),da.patient_id.to_numpy())})
    save_json(taskroot/"paired_ensemble_bootstrap.json",pairs)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    names=list(predictions_by_seed)
    for ax,key,title in zip(axes,["auroc","auprc","sensitivity"],["AUROC","Average precision","Sensitivity at frozen threshold"]):
        vals=[comparison.loc[(comparison.task==task)&(comparison.architecture==n),key].to_numpy() for n in names]
        ax.bar(range(len(names)),[v.mean() for v in vals],yerr=[v.std(ddof=1) for v in vals],color=[COLORS[n] for n in names],capsize=5,alpha=.8)
        for i,v in enumerate(vals): ax.scatter(np.repeat(i,len(v)),v,s=12,color="black",zorder=3)
        ax.set(xticks=range(len(names)),xticklabels=[DISPLAY[n].replace(" + ","\n") for n in names],title=title,ylim=(0,1)); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Individual models · mean ± sample SD across five seeds"); fig.tight_layout(); save_figure(fig,plots/"model_comparison")
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,names):
        runs=sorted(taskroot.glob(f"{name}__constrained__seed*"))
        for run in runs:
            h=pd.read_csv(run/"development/history.csv")
            ax.plot(h.epoch,h.dev_auroc,color=COLORS[name],alpha=.35)
            ax.plot(h.epoch,h.dev_auprc,color=COLORS[name],linestyle="--",alpha=.35)
        ax.set(title=DISPLAY[name],xlabel="Development epoch",ylabel="AUROC (solid) / AP (dashed)",ylim=(0,1)); ax.grid(alpha=.2)
    fig.suptitle("Internal development folds 7–8 · test fold excluded"); fig.tight_layout(); save_figure(fig,plots/"learning_curves")
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,names):
        for run in sorted(taskroot.glob(f"{name}__constrained__seed*")):
            h=pd.read_csv(run/"development/history.csv")
            ax.plot(h.epoch,h.train_loss,color=COLORS[name],alpha=.35)
            ax.plot(h.epoch,h.dev_bce,color="gray",linestyle="--",alpha=.35)
        ax.set(title=DISPLAY[name],xlabel="Development epoch",ylabel="Weighted training BCE / unweighted dev BCE"); ax.grid(alpha=.2)
    fig.suptitle("Loss histories · solid training / dashed development"); fig.tight_layout(); save_figure(fig,plots/"loss_curves")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in names:
        df=models[name]; p=df.probability.to_numpy(); y=df.label.to_numpy()
        order=np.argsort(-np.abs(p-.5)); coverage=[]; risk=[]
        for fraction in np.linspace(.1,1,19):
            ids=order[:max(1,int(len(y)*fraction))]; coverage.append(len(ids)/len(y))
            risk.append(np.mean((p[ids]>=thresholds[name])!=y[ids]))
        axes[0].plot(coverage,risk,color=COLORS[name],label=DISPLAY[name])
        for gender,marker in [(0,"o"),(1,"s")]:
            mask=df.sex==gender
            value=metrics(y[mask],p[mask],thresholds[name])["sensitivity"]
            axes[1].scatter(names.index(name)+(.08 if gender else -.08),value,marker=marker,color=COLORS[name],s=65)
    axes[0].set(xlabel="Retained fraction (score certainty ordering)",ylabel="Classification error",title="Selective risk · no deployment cutoff fitted"); axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(len(names)),xticklabels=[DISPLAY[n].replace(" + ","\n") for n in names],ylabel="Sensitivity",title="Sex subgroups · 0 circles / 1 squares",ylim=(0,1))
    fig.tight_layout(); save_figure(fig,plots/"coverage_and_subgroups")


def evaluate_suite(cache,root):
    root=Path(root)
    frozen=json.loads((root/"FROZEN_FOR_TEST.json").read_text())
    config=json.loads((root/"experiment.json").read_text())
    if digest(root/"experiment.json")!=frozen["configuration_sha256"]: raise ValueError("Experiment changed after freeze")
    save_json(root/"TEST_EVALUATION_STARTED.json",{"configuration_sha256":frozen["configuration_sha256"],"purpose":"One frozen held-out comparison; do not retune using these results."})
    rows=[]
    for run in frozen["runs"]:
        print("Final test evaluation",run,flush=True)
        rows.append(evaluate_run(cache,run,config["evaluation"]["patient_bootstrap_replicates"]))
    for task in config["tasks"]: rows.extend(classical_controls(cache,root,task))
    comparison=pd.DataFrame(rows); comparison.to_csv(root/"comparison_per_seed.csv",index=False)
    keys=["auroc","auprc","sensitivity","specificity","ppv","npv","f1","brier","parameters","selected_epochs","training_seconds"]
    neural=comparison[comparison.architecture.isin(config["architectures"])]
    summary=neural.groupby(["task","architecture"])[keys].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns]; summary.reset_index().to_csv(root/"comparison_summary.csv",index=False)
    for task in config["tasks"]: plot_task(root,task,comparison)
    save_json(root/"EVALUATION_COMPLETE.json",{"runs_evaluated":len(frozen["runs"]),"all_outputs_saved":True})
    return comparison
