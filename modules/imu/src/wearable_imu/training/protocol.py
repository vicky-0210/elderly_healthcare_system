"""Subject-grouped fitting and operating points, with one target per recording."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from wearable_imu.datasets.sisfall import CACHE
from wearable_imu.preprocessing.ipmv import view_names

RUN=Path("artifacts/imu/runs/sisfall_v1")
SEED=4907


def signature(spec):
    return hashlib.sha256((Path(CACHE)/"manifest.csv").read_bytes()+
        json.dumps(asdict(spec),sort_keys=True).encode()+b"sisfall-mil-v1"+
        (str(SEED).encode() if SEED!=4907 else b"")).hexdigest()


def partitions(folds,outer):
    val=(outer+1)%5
    return np.flatnonzero((folds!=outer)&(folds!=val)),np.flatnonzero(folds==val),np.flatnonzero(folds!=outer),np.flatnonzero(folds==outer)


def participant_weights(y,groups):
    weights=np.zeros(len(y))
    for c in (0,1):
        ids=np.unique(groups[y==c])
        if not len(ids): raise ValueError("Both trial classes required in validation")
        for s in ids:
            mask=(y==c)&(groups==s)
            weights[mask]=.5/(len(ids)*mask.sum())
    return weights


def operating_point(y,logits,groups):
    w=participant_weights(y,groups)
    order=np.argsort(-logits,kind="stable")
    levels=logits[order]
    end=np.r_[np.flatnonzero(np.diff(levels)!=0),len(y)-1]
    tp=np.r_[0,np.cumsum((w*y)[order])[end]]*2
    sp=1-2*np.r_[0,np.cumsum((w*(1-y))[order])[end]]
    thr=np.r_[np.nextafter(levels[0],np.inf),levels[end]]
    valid=sp>=.95-1e-10
    score=(tp+sp)/2
    i=int(np.argmax(np.where(valid,score,-1)))
    return float(score[i]),float(thr[i])


def calibration(y,s,groups):
    # Equal total subject weight, within the observed recording class mixture.
    w=np.array([1/np.sum(groups==g) for g in groups])
    w/=w.sum()
    def objective(ab):
        z=ab[0]*s+ab[1]
        return np.sum(w*(np.logaddexp(0,z)-y*z))+.001*((ab[0]-1)**2+ab[1]**2)
    fit=minimize(objective,[1.,0.],method="L-BFGS-B",bounds=[(.05,8.),(-12,12)])
    if not fit.success: raise ValueError(f"Calibration failed: {fit.message}")
    return list(map(float,fit.x))


def calibrated(logits,ab):
    return expit(np.asarray(logits,dtype=float)*ab[0]+ab[1])


def fit_scaler(tokens,rows,offsets,lengths,kind):
    idx=np.concatenate([np.arange(offsets[i],offsets[i]+lengths[i],20) for i in rows])
    sample=tokens[idx]
    if kind=="standard":
        center,scale=sample.mean(0,dtype=float),sample.std(0,dtype=float)
    elif kind=="robust":
        lo,center,hi=np.quantile(sample,[.25,.5,.75],axis=0)
        scale=hi-lo
        scale=np.where(scale>1e-6,scale,sample.std(0,dtype=float))
    else: raise ValueError(kind)
    return center.astype("f"),np.maximum(scale,1e-5).astype("f")


def normalize(x,center,scale):
    return np.arcsinh((x-center)/scale).astype("f",copy=False)


def make_data(spec,cache=CACHE):
    from wearable_imu.datasets.views import build_tokens,window_index
    m,t,names,offs,lengths=build_tokens(spec.settings,cache)
    selected=view_names(spec.view,names)
    x=np.ascontiguousarray(t[:,[names.index(k) for k in selected]])
    starts,g,bags,width=window_index(lengths,offs,spec.seconds,spec.overlap,spec.bag_size)
    return dict(m=m,x=x,names=selected,offs=offs,lengths=lengths,starts=starts,group=g,
        bags=bags,width=width,y=m.y.to_numpy(),subjects=m.subject.to_numpy(),folds=m.fold.to_numpy(),
        first=np.r_[0,np.cumsum(np.bincount(g))[:-1]])


def trial_max(d,rows,window_logits):
    # window_logits is in acquisition order restricted to these trials.
    sizes=np.bincount(d["group"])[rows]
    return np.maximum.reduceat(window_logits,np.r_[0,np.cumsum(sizes)[:-1]])


def write_output(folder,spec,d,outer,inner,test,inner_logits,test_logits,score,threshold,ab,extra):
    import pandas as pd
    folder.mkdir(parents=True,exist_ok=True)
    for ids,logits,name in ((inner,inner_logits,"validation"),(test,test_logits,"test")):
        wi=np.flatnonzero(np.isin(d["group"],ids))
        rec=d["m"].iloc[ids][["id","subject","activity","trial","older","y","seconds","fold"]].copy()
        bag=trial_max(d,ids,logits)
        rec["logit"],rec["p_fall"],rec["decision"]=bag,calibrated(bag,ab),(bag>=threshold).astype(int)
        rec.to_csv(folder/(name+".csv"),index=False)
        if name=="test":
            end=(d["starts"][wi]-d["offs"][d["group"][wi]]+d["width"])/20
            np.savez_compressed(folder/"test_windows.npz",trial=d["group"][wi],end=end,logit=logits.astype("f"))
    train,val,refit,held=partitions(d["folds"],outer)
    record=dict(spec=asdict(spec),outer=int(outer),signature=signature(spec),inner_score=score,threshold=threshold,
        calibration=ab,threshold_probability=float(calibrated(threshold,ab)),names=d["names"],width=int(d["width"]),
        train_subjects=sorted(set(d["subjects"][train])),val_subjects=sorted(set(d["subjects"][val])),
        refit_subjects=sorted(set(d["subjects"][refit])),held_subjects=sorted(set(d["subjects"][held])),**extra)
    marker=folder/"done.pending"
    marker.write_text(json.dumps(record,indent=2,allow_nan=False))
    marker.replace(folder/"done.json")


def complete(folder,spec):
    marker=folder/"done.json"
    if not marker.is_file(): return False
    if json.loads(marker.read_text())["signature"]!=signature(spec):
        raise RuntimeError(f"Different run provenance at {folder}; use another output directory")
    return True
