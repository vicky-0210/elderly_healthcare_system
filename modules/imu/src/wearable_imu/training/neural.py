"""A positive trial constrains the maximum of latent window logits."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from . import protocol as p
from wearable_imu.models.inertial import FallNet


def setup(d,ids,kind,device):
    center,scale=p.fit_scaler(d["x"],ids,d["offs"],d["lengths"],kind)
    # Use GPU-side transforms to avoid a second full CPU recording matrix.
    x=torch.as_tensor(d["x"],device=device)
    x=torch.asinh((x-torch.tensor(center,device=device))/torch.tensor(scale,device=device))
    return x,center,scale


def scan(model,x,starts,width):
    outputs=[]
    steps=torch.arange(width,device=x.device)
    model.eval()
    with torch.inference_mode():
        for block in np.array_split(starts,max(1,int(np.ceil(len(starts)/384)))):
            index=torch.as_tensor(block,device=x.device)[:,None]+steps
            outputs.append(model(x[index].transpose(1,2)).cpu().numpy())
    return np.concatenate(outputs)


def train_phase(spec,d,pack,ids,seed,epochs,val=None,teacher=None):
    x=pack[0]
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=FallNet(spec.model,d["names"],**spec.network).to(x.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0018,weight_decay=.001)
    starts=torch.as_tensor(d["starts"][d["bags"]],device=x.device)
    labels=torch.as_tensor(d["y"],dtype=torch.float32,device=x.device)
    positive=torch.tensor((d["y"][ids]==0).sum()/(d["y"][ids]==1).sum(),device=x.device,dtype=torch.float32)
    offsets=torch.arange(d["width"],device=x.device)
    if val is not None:
        validation_starts=d["starts"][np.isin(d["group"],val)]
    best,history=None,[]
    for epoch in range(1,epochs+1):
        net.train(); total=0.
        for batch in np.array_split(rng.permutation(ids),int(np.ceil(len(ids)/96))):
            idx=torch.as_tensor(batch,device=x.device)
            locations=(starts[idx,:,None]+offsets).reshape(-1,d["width"])
            windows=net(x[locations].transpose(1,2)).reshape(len(batch),-1)
            out=windows.amax(1)
            loss=F.binary_cross_entropy_with_logits(out,labels[idx],pos_weight=positive)
            if teacher is not None:
                # Smooth witness scores come only from an estimator fitted to these training subjects.
                distill=F.binary_cross_entropy_with_logits(windows/2,teacher[idx])
                loss=.5*loss+2*distill
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            total+=float(loss.detach())*len(batch)
        rec=dict(epoch=epoch,loss=total/len(ids))
        if not np.isfinite(rec["loss"]): raise FloatingPointError("Invalid MIL loss")
        if val is not None:
            scores=scan(net,x,validation_starts,d["width"])
            bag=p.trial_max(d,val,scores)
            value,threshold=p.operating_point(d["y"][val],bag,d["subjects"][val])
            rec["validation"]=value
            if best is None or value>best[0]+1e-9:
                best=(value,threshold,epoch,scores)
        history.append(rec)
        if val is not None and epoch-best[2]>=3: break
    return net,history,best


def save(path,net,pack,d,spec,threshold,ab):
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},spec=asdict(spec),
        names=d["names"],center=torch.from_numpy(pack[1]),scale=torch.from_numpy(pack[2]),
        width=int(d["width"]),threshold=threshold,calibration=ab),path)


def fit_teacher(spec,d,features,ids,seed,device):
    from dataclasses import replace
    from scipy.special import expit
    from .classical import fit,predict
    witness=fit(replace(spec,model="xgb"),d,features,ids,seed)
    selected=d["bags"][ids]
    unique,inv=np.unique(selected,return_inverse=True)
    targets=np.zeros(d["bags"].shape,dtype="f")
    targets[ids]=expit(predict(witness,features[unique])/2)[inv].reshape(selected.shape)
    return torch.as_tensor(targets,device=device)


def run_group(specs,root,device):
    d=p.make_data(specs[0])
    from .classical import make_features
    features=make_features(d) if any(s.name=="dm_distilled" for s in specs) else None
    for outer in range(5):
        todo=[s for s in specs if not p.complete(root/s.name/f"fold{outer}",s)]
        if not todo: continue
        train,val,refit,test=p.partitions(d["folds"],outer)
        dev_data=setup(d,train,specs[0].scaler,device)
        final_data=setup(d,refit,specs[0].scaler,device)
        for spec in todo:
            begin=time.perf_counter()
            folder=root/spec.name/f"fold{outer}"; folder.mkdir(parents=True,exist_ok=True)
            learn=fit_teacher(spec,d,features,train,p.SEED+outer,device) if spec.name=="dm_distilled" else None
            _,hist,(score,thr,epoch,vl)=train_phase(spec,d,dev_data,train,p.SEED+outer,spec.epochs,val,learn)
            ab=p.calibration(d["y"][val],p.trial_max(d,val,vl),d["subjects"][val])
            learn=fit_teacher(spec,d,features,refit,p.SEED+outer,device) if spec.name=="dm_distilled" else None
            net,refit_history,_=train_phase(spec,d,final_data,refit,p.SEED+outer,epoch,teacher=learn)
            test_starts=d["starts"][np.isin(d["group"],test)]
            logits=scan(net,final_data[0],test_starts,d["width"])
            path=folder/"model.pt"; save(path,net,final_data,d,spec,thr,ab)
            (folder/"history.json").write_text(json.dumps(dict(inner=hist,refit=refit_history)))
            p.write_output(folder,spec,d,outer,val,test,vl,logits,score,thr,ab,dict(epochs=epoch,
                parameters=sum(v.numel() for v in net.parameters()),bytes=path.stat().st_size,wall_seconds=time.perf_counter()-begin))
            print(f"{spec.name} fold {outer}: validation {score:.4f}, {epoch} epochs, {time.perf_counter()-begin:.1f}s",flush=True)
        del dev_data,final_data
