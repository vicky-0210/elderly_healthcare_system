"""Stress a fixed ten-minute contiguous PPG excerpt per person with frozen outer checkpoints."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,softmax
from wearable_ppg.preprocessing.optical import OpticalViews
from wearable_ppg.datasets.osasud import CACHE,RUN,RESULT
from wearable_ppg.datasets.windows import spans
from wearable_ppg.models.respiratory import RespiratoryNet
from wearable_ppg.evaluation.metrics import metrics


def main():
    torch.set_num_threads(2)
    name=json.loads((RESULT/"selection.json").read_text())["winner"]
    pieces=[]
    for r in pd.read_csv(RESULT/"folds.csv").itertuples():
        root=RUN/name/f"fold{r.fold}"; meta=json.loads((root/"done.json").read_text())
        model=RespiratoryNet(name).eval(); model.load_state_dict(torch.load(root/"model.pt",map_location="cpu",weights_only=True))
        with np.load(root/"scaler.npz") as p: center,scale=p["center"],p["scale"]
        with np.load(CACHE/f"p{r.patient:02}.npz") as f: raw,t,y,valid=(f[k] for k in ("raw","t","y","valid"))
        ix=max(spans(t,valid),key=len); ix=ix[(len(ix)-min(600,len(ix)))//2:][:600]  # centered in longest run, selected without labels
        x=raw[ix]; clock=t[ix]; label=y[ix]
        fs100=np.interp(np.arange(len(ix)*100)/100,np.arange(x.size)/80,x.ravel()).reshape(-1,100)
        rng=np.random.default_rng(502+r.patient); at=np.arange(fs100.size).reshape(fs100.shape)/100
        for condition in ("reference","haar_off","100hz","gain_offset","polarity_flip","contact_noise","dropouts"):
            fs=80 if condition in ("reference","haar_off","dropouts") else 100
            data=x if fs==80 else fs100; keep=np.ones(len(x),dtype=bool)
            if condition=="gain_offset": data=50000+90*data
            if condition=="polarity_flip": data=70000-90*data
            if condition=="contact_noise":
                data=50000+90*(data*(1+.10*np.sin(at/10))+6*np.sin(at*2*np.pi*.22)+rng.normal(0,2,at.shape))
            if condition=="dropouts": keep[(np.arange(len(keep))%100)<10]=False
            for run in spans(clock,keep):
                if len(run)<30: continue
                views=OpticalViews(fs,denoise=condition!="haar_off").push(data[run]); end=np.arange(29,len(run),5)
                idx=run[end[:,None]+np.arange(-4,1)].ravel()
                z=np.arcsinh((views[end[:,None]+np.arange(-29,1)]-center)/scale)
                with torch.inference_mode(): scores=model(torch.tensor(z)).numpy().reshape(-1,5)
                a,b=meta["calibration"]; prob=expit(a*scores[:,0]+b); sub=softmax(scores[:,1:],axis=1)
                frame=pd.DataFrame(dict(condition=condition,patient=r.patient,t=clock[idx],y=label[idx],p=prob,
                    pred=prob>=meta["threshold"],arrival=np.repeat(clock[run[end]]+1,5)))
                for j in range(4): frame[f"s{j}"]=sub[:,j]
                pieces.append(frame)
    table=pd.concat(pieces); out=[]
    for key,d in table.groupby("condition"):
        result,_,_=metrics(d.y.to_numpy(),d.p.to_numpy(),d[["s0","s1","s2","s3"]].to_numpy(),
            d.pred.to_numpy(),d.t.to_numpy(),d.patient.to_numpy(),d.arrival.to_numpy())
        ref=table[table.condition=="reference"][["patient","t","p","pred"]]
        pair=d.merge(ref,on=["patient","t"],suffixes=("","_ref"))
        out.append(dict(condition=key,common_seconds=len(pair),changed_decisions=int((pair.pred!=pair.pred_ref).sum()),
            mean_abs_probability_change=float(abs(pair.p-pair.p_ref).mean()),**result))
    pd.DataFrame(out).to_csv(RESULT/"transfer.csv",index=False)
    print(pd.DataFrame(out)[["condition","seconds","patient_ap","event_f1","changed_decisions"]].round(3))

if __name__=="__main__": main()
