from pathlib import Path
import numpy as np
import pandas as pd
from .osasud import CACHE,RESULT
from wearable_ppg.preprocessing.optical import OpticalViews,NAMES

STEP=5
CONTEXT=30


def spans(t,valid):
    idx=np.flatnonzero(valid)
    if not len(idx): return []
    cuts=np.flatnonzero((np.diff(idx)!=1)|(np.diff(t[idx])!=1))+1
    return np.split(idx,cuts)


def prepare_views():
    folder=CACHE/"views"; folder.mkdir(exist_ok=True)
    patients=[]; audit=[]
    for path in sorted(CACHE.glob("p*.npz")):
        pid=int(path.stem[1:]); f=np.load(path)
        t,y,raw,valid=(f[k] for k in ("t","y","raw","valid"))
        x=np.zeros((len(t),len(NAMES)),dtype="f"); ends=[]
        runs=spans(t,valid)
        for ix in runs:
            if len(ix)<CONTEXT: continue
            frontend=OpticalViews()
            for block in np.array_split(ix,max(1,int(np.ceil(len(ix)/2000)))):
                x[block]=frontend.push(raw[block])
            ends.extend(ix[CONTEXT-1::STEP])
        ends=np.asarray(ends,dtype=int)
        scorable=np.zeros(len(t),dtype=bool); scorable[ends[:,None]-np.arange(STEP)]=True
        np.savez(folder/path.name,x=x,t=t,y=y,ends=ends,valid=valid,scorable=scorable)
        audit.append(dict(patient=pid,windows=len(ends),scored_seconds=int(scorable.sum()),segments=len(runs),
                          positive_seconds=int((y[scorable]>0).sum()),mixed_seconds=int((y[scorable]==4).sum())))
        patients.append(pid)
    table=pd.DataFrame(audit)
    # Rank by eligible event prevalence then shuffle consecutive groups of five.
    # Every patient is tested once. Labels here are used solely for split stratification.
    rank=table.assign(prevalence=table.positive_seconds/table.scored_seconds).sort_values("prevalence").patient.to_numpy()
    rng=np.random.default_rng(2729); folds={}
    for block in np.array_split(rank,6):
        for k,pid in enumerate(rng.permutation(block)): folds[pid]=k
    table["fold"]=table.patient.map(folds)
    table.to_csv(RESULT/"folds.csv",index=False)
    print(table.sum(numeric_only=True).to_dict(),flush=True)


def load_views():
    frames=[]; endpoints=[]; seconds=[]; labels=[]; ids=[]; offset=0
    for r in pd.read_csv(RESULT/"folds.csv").itertuples():
        with np.load(CACHE/"views"/f"p{r.patient:02}.npz") as f:
            frames.append(f["x"]); labels.append(f["y"]); seconds.append(f["t"])
            ids.append(np.full(len(f["y"]),r.patient,dtype=np.int16)); endpoints.append(f["ends"]+offset)
            offset+=len(frames[-1])
    return tuple(np.concatenate(arr) for arr in (frames,labels,seconds,ids,endpoints))
