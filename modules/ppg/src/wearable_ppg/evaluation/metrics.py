"""Patient-level scores and one-to-one matching of contiguous labeled events."""
import numpy as np
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score,confusion_matrix


def contiguous(mask,t):
    idx=np.flatnonzero(mask)
    if not len(idx): return np.empty((0,2),dtype=np.int64)
    cut=np.flatnonzero((np.diff(idx)!=1)|(np.diff(t[idx])!=1))+1
    first=np.r_[0,cut]; last=np.r_[cut-1,len(idx)-1]
    return np.column_stack((t[idx[first]],t[idx[last]]+1))


def match_events(y,decision,t,arrival):
    true=contiguous(y,t); pred=contiguous(decision,t)
    pred=pred[np.diff(pred,axis=1).ravel()>=5]
    pairs=[]
    for i,(a,b) in enumerate(pred):
        first=np.searchsorted(true[:,1],a,side="right"); stop=np.searchsorted(true[:,0],b)
        for j in range(first,stop):
            c,d=true[j]
            iou=(min(b,d)-max(a,c))/(max(b,d)-min(a,c))
            if iou>=.3: pairs.append((-iou,i,j))
    used=set(); ref=set(); delays=[]
    for _,i,j in sorted(pairs):
        if i in used or j in ref: continue
        used.add(i); ref.add(j)
        confirm=np.searchsorted(t,pred[i,0]+4)
        delays.append(int(arrival[confirm]-true[j,0]))
    return len(used),len(pred)-len(used),len(true)-len(ref),delays


def patient_ap(y,p,ids):
    return float(np.mean([average_precision_score(y[ids==i]>0,p[ids==i]) for i in np.unique(ids)
                          if np.unique(y[ids==i]>0).size==2]))


def event_rows(y,d,t,ids,arrival):
    rows=[]; delays=[]
    for i in np.unique(ids):
        g=ids==i; tp,fp,fn,delay=match_events(y[g]>0,d[g],t[g],arrival[g])
        rows.append(dict(patient=int(i),event_tp=tp,event_fp=fp,event_fn=fn,event_f1=2*tp/max(1,2*tp+fp+fn),
                         hours=float(g.sum()/3600),true_events=tp+fn))
        delays.extend(delay)
    return rows,delays


def choose_threshold(y,p,t,ids,arrival):
    best=(-1,0)
    for thr in np.unique(np.quantile(p,np.linspace(.40,.99,26))):
        rows,_=event_rows(y,p>=thr,t,ids,arrival)
        value=np.mean([r["event_f1"] for r in rows if r["true_events"]>0])
        if (value,thr)>best: best=value,thr
    return float(best[1])


def metrics(y,p,sub,d,t,ids,arrival):
    from sklearn.metrics import precision_score,recall_score
    b=y>0; pred=np.where(d,1+sub.argmax(1),0)
    rows,delay=event_rows(y,d,t,ids,arrival)
    tp,fp,fn=[sum(r[k] for r in rows) for k in ("event_tp","event_fp","event_fn")]
    tn,wfp,wfn,wtp=confusion_matrix(b,d,labels=[False,True]).ravel()
    return dict(seconds=len(b),participants=len(set(ids)),prevalence=float(b.mean()),patient_ap=patient_ap(y,p,ids),
        ap=average_precision_score(b,p),auc=roc_auc_score(b,p),precision=precision_score(b,d,zero_division=0),
        recall=recall_score(b,d),specificity=float(tn/(tn+wfp)),binary_f1=f1_score(b,d),brier=float(np.mean((p-b)**2)),
        five_macro_f1=f1_score(y,pred,labels=list(range(5)),average="macro",zero_division=0),event_tp=tp,event_fp=fp,event_fn=fn,
        event_f1=2*tp/max(1,2*tp+fp+fn),event_precision=tp/max(1,tp+fp),event_recall=tp/max(1,tp+fn),
        patient_event_f1=float(np.mean([r["event_f1"] for r in rows if r["true_events"]>0])),
        false_events_per_hour=fp/(len(b)/3600),median_confirmation_delay_s=float(np.median(delay)) if delay else None),rows,pred
