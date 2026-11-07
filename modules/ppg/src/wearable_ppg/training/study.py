import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit,softmax
from sklearn.linear_model import LogisticRegression
from wearable_ppg.datasets.osasud import RESULT,RUN
from wearable_ppg.datasets.windows import load_views
from wearable_ppg.evaluation.metrics import patient_ap,choose_threshold
from wearable_ppg.models.respiratory import RespiratoryNet

SEED=6811
EPOCHS=10


def seconds(ends,delay=0):
    return ends[:,None]+np.arange(-4,1)-delay


def norm_fit(x,ends):
    # One eligible endpoint per ten seconds approximately, all training patients.
    sample=x[ends[::2]]
    q=np.percentile(sample,[25,50,75],axis=0)
    spread=q[2]-q[0]
    return q[1].astype("f"),np.maximum(np.where(spread>.001,spread,sample.std(0)),.01).astype("f")


def build_input(x,ends,window,device):
    idx=torch.as_tensor(ends,device=device)[:,None]+torch.arange(1-window,1,device=device)
    return x[idx]


def infer(net,x,ends,window):
    net.eval(); out=[]
    with torch.inference_mode():
        for part in np.array_split(ends,max(1,int(np.ceil(len(ends)/512)))):
            out.append(net(build_input(x,part,window,x.device)).cpu().numpy())
    return np.concatenate(out).reshape(-1,5)


def sample_train(ends,ids,y,rng,delay=0):
    samples=[]
    for i in np.unique(ids[ends]):
        eligible=ends[ids[ends]==i]
        positive=eligible[np.any(y[seconds(eligible,delay)]>0,axis=1)]
        samples.append(rng.choice(eligible,400,replace=True))
        samples.append(rng.choice(positive if len(positive) else eligible,240,replace=True))
    return rng.permutation(np.concatenate(samples))


def train_net(name,x,y,t,ids,train,valid,epochs,device,seed):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    window={"context_10":10,"context_20":20}.get(name,30)
    delay=10 if name=="lag10" else 0
    center,scale=norm_fit(x,train)
    xt=torch.asinh((torch.as_tensor(x,device=device)-torch.tensor(center,device=device))/torch.tensor(scale,device=device))
    net=RespiratoryNet(name).to(device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.005)
    counts=np.bincount(y[seconds(train,delay)].ravel(),minlength=5)[1:]
    weights=torch.tensor(np.clip(np.sqrt(counts.max()/np.maximum(1,counts)),1,5),dtype=torch.float32,device=device)
    best=-np.inf; history=[]; state=None; picked=0
    for epoch in range(1,epochs+1):
        net.train(); losses=[]
        sampled=sample_train(train,ids,y,rng,delay)
        for ends in np.array_split(sampled,int(np.ceil(len(sampled)/192))):
            inp=build_input(xt,ends,window,device)
            labels=torch.as_tensor(y[seconds(ends,delay)],device=device,dtype=torch.long)
            out=net(inp); mask=labels>0
            loss=F.binary_cross_entropy_with_logits(out[:,:,0],mask.float())
            if mask.any(): loss=loss+.35*F.cross_entropy(out[:,:,1:][mask],labels[mask]-1,weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),2); opt.step()
            losses.append(float(loss.detach()))
        entry=dict(epoch=epoch,loss=float(np.mean(losses)))
        if valid is not None:
            scores=infer(net,xt,valid,window); idx=seconds(valid,delay).ravel()
            value=patient_ap(y[idx],scores[:,0],ids[idx])
            entry["patient_ap"]=value
            if value>best+.0001:
                best=value; state=copy.deepcopy(net.state_dict()); picked=epoch
        history.append(entry)
        if valid is not None and epoch-picked>=3: break
    if state is not None: net.load_state_dict(state)
    return net,xt,window,center,scale,history,picked or epochs


def calibration(y,z):
    lr=LogisticRegression(C=10,max_iter=150)
    lr.fit(z[:,None],y>0)
    return [float(max(.00001,lr.coef_[0,0])),float(lr.intercept_[0])]


def predictions(folder,key,net,xt,ends,window,y,t,ids,ab=None,thr=None):
    raw=infer(net,xt,ends,window); idx=seconds(ends,10 if net.name=="lag10" else 0).ravel()
    # Every score is available at the exclusive end of its five-second output block.
    arrival=np.repeat(t[ends]+1,5)
    if ab is None: ab=calibration(y[idx],raw[:,0])
    p=expit(ab[0]*raw[:,0]+ab[1]); sub=softmax(raw[:,1:],axis=1)
    if thr is None: thr=choose_threshold(y[idx],p,t[idx],ids[idx],arrival)
    np.savez_compressed(folder/(key+".npz"),patient=ids[idx],t=t[idx],arrival=arrival,y=y[idx],p=p,
        subtype=sub,decision=p>=thr,logit=raw[:,0])
    return ab,thr,patient_ap(y[idx],p,ids[idx])


def run(names,device):
    torch.set_num_threads(2)
    x,y,t,ids,ends=load_views(); cohort=pd.read_csv(RESULT/"folds.csv")
    for k in range(5):
        valid_ids=cohort.loc[cohort.fold==(k+1)%5,"patient"].tolist()
        test_ids=cohort.loc[cohort.fold==k,"patient"].tolist()
        train_ids=sorted(set(cohort.patient)-set(valid_ids)-set(test_ids))
        train,val,test=[ends[np.isin(ids[ends],group)] for group in (train_ids,valid_ids,test_ids)]
        for name in names:
            folder=RUN/name/f"fold{k}"; folder.mkdir(parents=True,exist_ok=True)
            if (folder/"done.json").exists(): continue
            start=time.perf_counter()
            net,xt,window,center,scale,hist,epoch=train_net(name,x,y,t,ids,train,val,EPOCHS,device,SEED+k)
            ab,thr,score=predictions(folder,"validation",net,xt,val,window,y,t,ids)
            predictions(folder,"test",net,xt,test,window,y,t,ids,ab,thr)
            state={k:v.detach().cpu() for k,v in net.state_dict().items()}
            torch.save(state,folder/"model.pt"); np.savez(folder/"scaler.npz",center=center,scale=scale)
            meta=dict(name=name,fold=k,train=train_ids,val=valid_ids,test=test_ids,context=window,epoch=epoch,
                calibration=ab,threshold=thr,val_patient_ap=score,parameters=sum(p.numel() for p in net.parameters()),
                wall_seconds=time.perf_counter()-start,history=hist)
            marker=folder/"done.tmp"; marker.write_text(json.dumps(meta,indent=2)); marker.replace(folder/"done.json")
            print(f"{name} fold {k}: validation patient AP {score:.3f}, epoch {epoch}, {meta['wall_seconds']:.1f}s",flush=True)
