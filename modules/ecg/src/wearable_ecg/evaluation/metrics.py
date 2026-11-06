"""Calibration, frozen operating points and patient-cluster uncertainty."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             f1_score, brier_score_loss, log_loss, roc_curve)


def choose_threshold(y,p,specificity=.90):
    """Highest sensitivity satisfying empirical specificity; handle ties exactly."""
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    valid=np.flatnonzero(fpr<=1-specificity+1e-12)
    best=valid[np.argmax(tpr[valid])]
    value=float(thresholds[best])
    if not np.isfinite(value): value=float(np.nextafter(np.max(p),np.inf))
    return value


def choose_sensitive_threshold(y,p,sensitivity=.95):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    valid=np.flatnonzero(tpr>=sensitivity)
    i=valid[np.argmin(fpr[valid])]
    return float(thresholds[i])


def fit_calibration(logits,y):
    # Positive slope preserves ranking; intercept handles class-weight shift.
    def objective(ab):
        z=np.exp(ab[0])*np.asarray(logits)+ab[1]
        return np.mean(np.logaddexp(0,z)-np.asarray(y)*z)+1e-4*ab[0]**2
    result=minimize(objective,[0.,0.],method="L-BFGS-B",bounds=[(-4,4),(-10,10)])
    if not result.success or not np.isfinite(result.fun): raise RuntimeError("Calibration failed")
    return {"slope":float(np.exp(result.x[0])),"intercept":float(result.x[1]),"fit_partition":"fold_9"}


def calibrated(logits,params): return expit(params["slope"]*np.asarray(logits)+params["intercept"])


def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    def div(a,b): return float(a/b) if b else None
    result={"n":len(y),"positive":int(y.sum()),"prevalence":float(np.mean(y)),
            "auroc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "auprc":float(average_precision_score(y,p)) if y.sum() else None,
            "sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),"ppv":div(tp,tp+fp),
            "npv":div(tn,tn+fn),"f1":float(f1_score(y,p>=threshold,zero_division=0)),
            "accuracy":float(np.mean(y==(p>=threshold))),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"threshold":float(threshold)}
    bins=np.minimum((p*10).astype(int),9)
    result["ece_10bins"]=float(sum(np.mean(bins==i)*abs(np.mean(y[bins==i])-np.mean(p[bins==i])) for i in range(10) if np.any(bins==i)))
    return result


def bootstrap(y,p,patients,threshold,n_boot=1000,seed=2026):
    y=np.asarray(y); p=np.asarray(p); patients=np.asarray(patients)
    unique=np.unique(patients); groups={k:np.flatnonzero(patients==k) for k in unique}
    rng=np.random.default_rng(seed)
    keys=["auroc","auprc","sensitivity","specificity","ppv","npv","f1","brier"]
    samples={k:[] for k in keys}
    for _ in range(n_boot):
        ids=np.concatenate([groups[k] for k in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ids]))<2: continue
        m=metrics(y[ids],p[ids],threshold)
        for key in keys:
            if m[key] is not None: samples[key].append(m[key])
    return {k:{"low":float(np.quantile(v,.025)),"high":float(np.quantile(v,.975)),"replicates":len(v)} for k,v in samples.items() if v}


def paired_bootstrap(y,pa,pb,patients,n_boot=1000,seed=2026):
    unique=np.unique(patients); groups={k:np.flatnonzero(np.asarray(patients)==k) for k in unique}
    rng=np.random.default_rng(seed); auc=[]; ap=[]
    for _ in range(n_boot):
        ids=np.concatenate([groups[k] for k in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ids]))<2: continue
        auc.append(roc_auc_score(y[ids],pa[ids])-roc_auc_score(y[ids],pb[ids]))
        ap.append(average_precision_score(y[ids],pa[ids])-average_precision_score(y[ids],pb[ids]))
    return {"auroc_delta":[float(np.mean(auc)),*np.quantile(auc,[.025,.975]).tolist()],
            "auprc_delta":[float(np.mean(ap)),*np.quantile(ap,[.025,.975]).tolist()]}
