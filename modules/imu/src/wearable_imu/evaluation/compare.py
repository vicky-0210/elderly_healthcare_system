"""Aggregate complete subject-disjoint runs and evaluate the prespecified selectors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss)
from sklearn.tree import DecisionTreeClassifier
from wearable_imu.inference.alerts import AlertPolicy
from wearable_imu.training.catalogue import catalogue
from wearable_imu.training.protocol import complete,calibrated
from wearable_imu.datasets.sisfall import CACHE


def metrics(r):
    y,d=r.y.to_numpy(),r.decision.to_numpy()
    tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel().tolist()
    rates=r.assign(correct=(y==d)).groupby(["y","subject"]).correct.mean()
    def ratio(a,b): return a/b if b else np.nan
    se,sp=ratio(tp,tp+fn),ratio(tn,tn+fp)
    mse=rates.get(1); msp=rates.get(0)
    mse=float(mse.mean()) if mse is not None else np.nan
    msp=float(msp.mean()) if msp is not None else np.nan
    result=dict(trials=len(r),participants=r.subject.nunique(),tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=(tp+tn)/len(r),precision=ratio(tp,tp+fp),recall=se,specificity=sp,
        f1=ratio(2*tp,2*tp+fp+fn),subject_sensitivity=mse,subject_specificity=msp,subject_ba=(mse+msp)/2,
        auc=roc_auc_score(y,r.p_fall) if np.unique(y).size==2 else np.nan,
        ap=average_precision_score(y,r.p_fall) if np.any(y) else np.nan,brier=brier_score_loss(y,r.p_fall,pos_label=1))
    if "alerts" in r:
        a=r[y==0]; hours=a.seconds.sum()/3600; effective=a.monitored.sum()/3600
        result.update(adl_hours=hours,monitored_hours=effective,false_alerts=int(a.alerts.sum()),
            fa_per_hour=ratio(a.alerts.sum(),hours),fa_per_monitored_hour=ratio(a.alerts.sum(),effective))
    return result


def decorate(r,folder,meta,m):
    windows=np.load(folder/"test_windows.npz"); idx=windows["trial"]; logits=windows["logit"]; end=windows["end"]
    starts=np.r_[0,np.flatnonzero(np.diff(idx))+1]; stops=np.r_[starts[1:],len(idx)]
    if not np.array_equal(m.id.to_numpy()[idx[starts]],r.id): raise ValueError("Trial/window IDs disagree")
    alerts=[]; fixed=[]; monitored=[]
    for start,stop in zip(starts,stops):
        z=logits[start:stop]; t=end[start:stop]
        equal=z[np.rint(np.linspace(0,len(z)-1,8)).astype(int)]
        fixed.append(float(equal.max()))
        policy=AlertPolicy(meta["threshold"])
        alerts.append(sum(policy.step(s,h) for s,h in zip(z,t)))
        monitored.append(t[-1]-t[0])
    r=r.copy(); r["alerts"]=alerts; r["monitored"]=monitored; r["fixed_logit"]=fixed
    r["fixed_decision"]=(np.asarray(fixed)>=meta["threshold"]).astype(int)
    r["fixed_p"]=calibrated(fixed,meta["calibration"]); r["model"]=meta["spec"]["name"]
    return r


def select(rows):
    cutoff=rows.inner_score.max()-.005
    return rows[rows.inner_score>=cutoff].sort_values(["bytes","model"]).iloc[0].model


def bootstrap(r,other=None):
    def rates(x):
        return x.assign(correct=x.y==x.decision).pivot_table(index="subject",columns="y",values="correct").sort_index()
    a=rates(r)
    if other is not None: a=a-rates(other).reindex(a.index)
    rng=np.random.default_rng(927)
    draws=a.to_numpy()[rng.integers(0,len(a),(2000,len(a)))]
    scores=np.nanmean(draws,axis=1).mean(1)
    return np.nanquantile(scores,[.025,.975]).tolist()


def fixed(r):
    return r.assign(decision=r.fixed_decision,p_fall=r.fixed_p)


def compare(root,out):
    specs=catalogue(); m=pd.read_csv(CACHE/"manifest.csv")
    out.mkdir(parents=True,exist_ok=True)
    all_oof={}; fold_stats=[]; summaries=[]
    for name,spec in specs.items():
        trials=[]; scores=[]
        for fold in range(5):
            f=root/name/f"fold{fold}"
            if not complete(f,spec): raise RuntimeError(f"Incomplete run: {f}")
            meta=json.loads((f/"done.json").read_text())
            if set(meta["held_subjects"])&set(meta["refit_subjects"]) or set(meta["train_subjects"])&set(meta["val_subjects"]):
                raise AssertionError("Participant leakage")
            r=decorate(pd.read_csv(f/"test.csv"),f,meta,m)
            if set(r.subject)!=set(meta["held_subjects"]): raise AssertionError("Wrong outer participants")
            trials.append(r); scores.append(meta["inner_score"])
            fold_stats.append(dict(model=name,role=spec.role,fold=fold,inner_score=meta["inner_score"],bytes=meta["bytes"],
                threshold=meta["threshold"],epochs=meta.get("epochs"),parameters=meta["parameters"],wall_seconds=meta["wall_seconds"],**metrics(r)))
        r=pd.concat(trials,ignore_index=True)
        if set(r.id)!=set(m.id) or r.id.duplicated().any(): raise AssertionError("Incomplete test coverage")
        all_oof[name]=r
        summaries.append(dict(model=name,role=spec.role,inner_score=np.mean(scores),fixed8_subject_ba=metrics(fixed(r))["subject_ba"],**metrics(r)))
    folds=pd.DataFrame(fold_stats); folds.to_csv(out/"fold_metrics.csv",index=False)
    score=pd.DataFrame(summaries).merge(folds.groupby("model",as_index=False)[["bytes","parameters","wall_seconds"]].mean(),on="model")
    score.sort_values("inner_score",ascending=False).to_csv(out/"comparison.csv",index=False)
    pd.concat(all_oof.values(),ignore_index=True).to_csv(out/"all_oof_trials.csv.gz",index=False,compression="gzip")
    prim=score[score.role=="primary"]
    menus={"all":score,"primary":prim,"neural":prim[~prim.model.isin(["xgb","rf","svm"])]}
    selections={}; choice_records=[]; panels={}
    for menu,rows in menus.items():
        winner=select(rows); selected=[]
        for fold in range(5):
            candidates=folds[(folds.fold==fold)&folds.model.isin(rows.model)]
            picked=select(candidates)
            trial=all_oof[picked]; selected.append(trial[trial.fold==fold])
            choice_records.append(dict(menu=menu,outer=fold,model=picked))
        pooled=pd.concat(selected,ignore_index=True)
        panels["nested_"+menu]=pooled
        panels["fixed_"+menu]=all_oof[winner]
        selections[menu]=dict(final=winner,nested=metrics(pooled),cluster95=bootstrap(pooled),
            fixed=metrics(all_oof[winner]),fixed8=metrics(fixed(all_oof[winner])))
    pd.DataFrame(choice_records).to_csv(out/"outer_choices.csv",index=False)
    panels["dm_tcn"]=all_oof["dm_tcn"]
    selected=pd.concat([r.assign(policy=k) for k,r in panels.items()],ignore_index=True)
    selected.to_csv(out/"selected_trials.csv.gz",index=False,compression="gzip")
    summaries=[]
    for policy,r in panels.items():
        for column in ("activity","subject","older"):
            for key,part in r.groupby(column):
                summaries.append(dict(policy=policy,stratum=column,value=key,**metrics(part)))
    pd.DataFrame(summaries).to_csv(out/"subgroups.csv",index=False)
    controls=[]
    for fold in range(5):
        train=m[m.fold!=fold]; test=m[m.fold==fold].copy()
        tree=DecisionTreeClassifier(max_depth=3,random_state=42).fit(train[["seconds"]],train.y)
        test["p_fall"]=tree.predict_proba(test[["seconds"]])[:,1]; test["decision"]=(test.p_fall>=.5).astype(int)
        controls.append(test)
    selections["duration_only"]=metrics(pd.concat(controls))
    selections["paired_subject_ba_95_nested_neural_minus_dm"]=bootstrap(panels["nested_neural"],panels["dm_tcn"])
    (out/"selection.json").write_text(json.dumps(selections,indent=2,allow_nan=False))
    from .figures import figures
    figures(score,folds,panels,m,out)
    return selections
