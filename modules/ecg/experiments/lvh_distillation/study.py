"""PTB-XL morphology distillation study. Run from the team repository root."""
import argparse, ast, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy import signal, ndimage
from scipy.special import expit
from sklearn.metrics import matthews_corrcoef
from wearable_ecg.preprocessing.pipeline import landmarks
from wearable_ecg.evaluation.metrics import metrics, choose_threshold, fit_calibration, calibrated
from wearable_ecg.runtime import seed_everything, save_json
from wearable_ecg.training.runner import train_run, predict_logits
from wearable_ecg.models.backbones import make_model
from network import MorphNet

C=Path("data/ecg/processed/ptbxl_360hz_v1")
A=Path("artifacts/ecg/runs/lvh_distill_v1")
R=Path("research/results/lvh_distill_v1")
OLD=A.parent/"20261105_lvh_comparison"
DEV=[7,8]
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BASE={"hz":360,"pp":"constrained","loss":"weighted","epochs":7,"compute":"cuda_amp_v1"}

def array(name):
    return np.load(C/(name+".npy"),mmap_mode="r")

def baseline():
    cfg=json.loads((OLD/"experiment.json").read_text())["training"]
    path=train_run(C,A,"lvh_vs_rest","cnn",42,cfg)
    def best(folder):
        epoch=json.loads((folder/"COMPLETE.json").read_text())["development"]["best_epoch"]
        return pd.read_csv(folder/"development/history.csv").iloc[epoch-1]
    a,b=best(path),best(OLD/"lvh_vs_rest"/path.name)
    delta={c:float(a[c]-b[c]) for c in ("dev_auprc","dev_sensitivity_at_90spec")}
    ok=abs(delta["dev_auprc"])<=.005 and abs(delta["dev_sensitivity_at_90spec"])<=.02
    save_json(A/"E0.json",dict(passed=ok,delta=delta,reproduced=a.to_dict(),reference=b.to_dict()))
    if not ok: raise RuntimeError(f"E0 outside reproduction tolerance: {delta}")

def processed(x,kind):
    if kind=="drift": return x-ndimage.uniform_filter1d(x,size=217,axis=-1,mode="reflect")
    if kind=="mains":
        z=np.fft.rfft(x); f=np.fft.rfftfreq(x.shape[-1],1/360)
        for hz in (50,60):
            ix=np.abs(f-hz)<.15
            use=(abs(z[:,ix])**2).mean(1)>8*(abs(z[:,(abs(f-hz)>1)&(abs(f-hz)<4)])**2).mean(1)
            z[np.ix_(use,ix)]=0
        return np.fft.irfft(z,n=x.shape[-1]).astype("f")
    return x

def prepare():
    A.mkdir(parents=True,exist_ok=True); R.mkdir(parents=True,exist_ok=True)
    if (A/"prepared.json").exists(): return
    raw=array("minimal_waves")
    for name in ("drift","mains","hz100"):
        width=1000 if name=="hz100" else 3600
        out=np.lib.format.open_memmap(A/(name+".npy"),mode="w+",dtype="f",shape=(len(raw),width))
        for i in range(0,len(raw),128):
            sl=slice(i,i+128)
            out[sl]=signal.resample_poly(array("lead_ii_500hz_mv")[sl],1,5,axis=-1) if name=="hz100" else processed(raw[sl],name)
        del out
    x=array("constrained_waves")
    beats=np.zeros((len(x),144),dtype="f")
    for i,v in enumerate(x):
        p=landmarks(v); p=p[(p>=54)&(p+90<=len(v))]
        if len(p): beats[i]=np.median(v[p[:,None]+np.arange(-54,90)],axis=0)
    np.save(A/"beats.npy",beats)
    rows=[]
    m=pd.read_csv(C/"lvh_vs_rest_manifest.csv")
    for i in m[m.strat_fold.isin(DEV)].row_index:
        v=raw[i]; p=landmarks(v)
        if not len(p): continue
        ix=p[:,None]+np.arange(-28,37)
        base=np.ptp(v[ix],axis=1).mean()
        for key,source in DATA.items():
            if key=="hz100": continue
            w=source()[i]
            rows.append([key,np.ptp(w[ix],axis=1).mean()-base,np.sqrt(np.mean((w[ix]-v[ix])**2)),np.mean(np.diff(w)==0),np.mean(abs(w)>5),np.mean(~np.isfinite(w)),np.mean((w==w.min())|(w==w.max()))])
    pd.DataFrame(rows,columns=["variant","qrs_span_delta_mv","qrs_change_rms_mv","constant_pair_fraction","over_5mv_fraction","missing_fraction","extrema_dwell_proxy"]).groupby("variant").mean().to_csv(R/"preprocessing.csv")
    save_json(A/"prepared.json",{"complete":True})

DATA={k:(lambda k=k:array(k+"_waves")) for k in ("constrained","minimal","conventional_fixed")}
DATA.update({k:(lambda k=k:np.load(A/(k+".npy"),mmap_mode="r")) for k in ("mains","drift","hz100")})

def patient(frame,z):
    return pd.DataFrame(dict(id=frame.patient_id.to_numpy(),y=frame.label.to_numpy(),z=z)).groupby("id").max()

def measure(y,p,thr):
    m=metrics(y,p,thr)
    m.update(balanced_accuracy=(m["specificity"]+m["sensitivity"])/2,mcc=matthews_corrcoef(y,p>=thr))
    return m

def devscore(frame,z):
    scores=[]
    for k in DEV:
        inside=frame.strat_fold.to_numpy()==k
        a,b=patient(frame[~inside],z[~inside]),patient(frame[inside],z[inside])
        cal=fit_calibration(a.z,a.y); p=calibrated(b.z,cal)
        m=measure(b.y,p,choose_threshold(b.y,p,.9))
        scores.append([m["auprc"],m["sensitivity"],m["f1"],m["brier"],m["ece_10bins"]])
    return dict(zip(("ap","sens90","f1","brier","ece"),np.mean(scores,axis=0)))

def choose(rows,base=None):
    valid=[r for r in rows if (base is None or r["brier"]<=base["brier"]+.01) and r.get("quality_ap_drop",0)<=.015]
    if not valid: return None
    top=max(r["ap"] for r in valid)
    return max((r for r in valid if r["ap"]>=top-.005),key=lambda r:(r["sens90"],r["f1"],-r["params"],-r.get("ms",0)))

class Study:
    def __init__(self):
        m=pd.read_csv(C/"manifest.csv")
        self.m=m
        self.eligible=pd.read_csv(C/"lvh_vs_rest_manifest.csv")
        self.codes=m.scp_codes.map(ast.literal_eval)
        source=Path(json.loads((C/"COMPLETE.json").read_text())["source"])
        scp=pd.read_csv(source/"scp_statements.csv",index_col=0)
        groups=[set(scp.index[scp.diagnostic_class==key])-{"LVH","VCLVH"} for key in ("STTC","CD","HYP","MI")]
        self.aux=np.array([[max([v for k,v in d.items() if k in g],default=-1) for g in groups] for d in self.codes])
        self.hard=(self.aux>=0).any(1)|self.codes.map(lambda d:bool(set(d)&{"VCLVH","ABQRS","HVOLT"})).to_numpy()
        self.beats=np.load(A/"beats.npy",mmap_mode="r")
        self.rows=[]; self.teacher=None

    def train(self,name,cfg,final=False):
        c=BASE|cfg; folder=A/("refit" if final else "development_amp")/name
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"done.json").exists():
            row=json.loads((folder/"done.json").read_text())
            if row["config"]!=c: raise RuntimeError("Stale configuration "+name)
            return row
        seed_everything(42); frame=self.eligible
        torch.backends.cudnn.deterministic=False
        scaler=torch.amp.GradScaler("cuda",enabled=DEVICE=="cuda")
        tr=frame[frame.strat_fold<=(8 if final else 6)].row_index.to_numpy()
        if c.get("exclude"): tr=tr[(self.m.label[tr].values==0)|(self.m.lvh_confidence[tr].values>=c["exclude"])]
        va=frame[frame.strat_fold.isin([9,10] if final else DEV)].row_index.to_numpy()
        teacher=bool(c.get("teacher"))
        data=array("teacher_12lead_360hz_mv") if teacher else DATA[c["pp"]]()
        feat=array("constrained_features")
        model=MorphNet(c).to(DEVICE)
        model.center.copy_(torch.tensor(feat[tr].mean(0))); model.scale.copy_(torch.tensor(feat[tr].std(0).clip(.01)))
        if c.get("robust"):
            q=np.quantile(data[tr[::15],::12],[.25,.5,.75])
            model.offset.fill_(q[1]); model.spread.fill_(max(.01,q[2]-q[0]))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        rng=np.random.default_rng(42)
        y=self.m.label.to_numpy(); pi=y[tr].mean()
        def batch(ids):
            v=torch.as_tensor(np.asarray(data[ids],dtype="f"),device=DEVICE)
            return (v if teacher else v[:,None],torch.tensor(feat[ids],device=DEVICE),torch.tensor(self.beats[ids,None,:],device=DEVICE))
        def tensor(x): return torch.as_tensor(np.asarray(x,dtype="f"),device=DEVICE)
        if c.get("ssl"):
            for _ in range(2):
                for ids in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/128))):
                    x,_,_=batch(ids); tokens=x.unfold(-1,model.patch,model.patch).squeeze(1)
                    masked=torch.rand(tokens.shape[:2],device=DEVICE)<.25
                    inp=tokens.masked_fill(masked[:,:,None],0).reshape_as(x)
                    loss=F.mse_loss(model.rebuild(model.encode(inp)).transpose(1,2)[masked],tokens[masked])
                    opt.zero_grad(); loss.backward(); opt.step()
        if c.get("kd") or c.get("rep"):
            target=np.load(A/("refit" if final else "development_amp")/self.teacher/"targets.npz")
            tz,latent=target["logit"],target["latent"]
        history=[]
        hardness=np.ones(len(tr))
        for ep in range(c["epochs"]):
            model.train()
            order=rng.permutation(tr)
            method=c["loss"]
            if method=="balanced":
                w=np.where(y[tr],1/pi,1/(1-pi)); order=rng.choice(tr,len(tr),p=w/w.sum())
            if method=="mining":
                w=1+((1-y[tr])*self.hard[tr]*hardness); order=rng.choice(tr,len(tr),p=w/w.sum())
            total=0.
            for ids in np.array_split(order,int(np.ceil(len(order)/128))):
                x,f,beat=batch(ids)
                if c.get("augment"): x=corrupt(x,c["hz"])
                mask=None
                if teacher:
                    mask=torch.rand(len(ids),12,device=DEVICE)>.12; mask[:,1]=True
                with torch.autocast(device_type=DEVICE,dtype=torch.float16,enabled=DEVICE=="cuda"):
                    z,h,aux=model(x,f,beat,mask)
                z,h,aux=z.float(),h.float(),aux.float()
                target=tensor(y[ids]); conf=tensor(np.nan_to_num(self.m.lvh_confidence.values[ids],nan=100)/100)
                sup=target*conf if c.get("label")=="soft" else target
                logit=z+np.log(pi/(1-pi)) if method=="adjust" else z
                b=F.binary_cross_entropy_with_logits(logit,sup,reduction="none")
                if method in ("weighted","mining"): b=b*(1+sup*((1-pi)/pi-1))
                if method in ("focal","cbfocal"):
                    p=z.sigmoid(); b=b*(sup*(1-p)+(1-sup)*p)**2
                    if method=="cbfocal": b=b*(1+sup*((1-.9999**((1-pi)*len(tr)))/(1-.9999**(pi*len(tr)))-1))
                if c.get("label")=="weighted": b=b*conf
                loss=b.mean()
                if c.get("kd") or c.get("rep"):
                    T=c["T"]; t=tensor(tz[ids]); gate=(2*t.sigmoid()-1).abs() if c.get("gate") else 1
                    loss=loss+c["kd"]*T*T*(gate*F.binary_cross_entropy_with_logits(z/T,(t/T).sigmoid(),reduction="none")).mean()
                    if c.get("rep"): loss=loss+c["rep"]*(gate*(1-F.cosine_similarity(h,tensor(latent[ids])))).mean()
                if c.get("aux"):
                    values=tensor(self.aux[ids][:,c["aux"]]); known=(values<0)|(values>=50)
                    term=F.binary_cross_entropy_with_logits(aux[:,c["aux"]],(values>=50).float(),reduction="none")
                    loss=loss+.2*(term*known).sum()/known.sum().clamp_min(1)
                opt.zero_grad();scaler.scale(loss).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),2);scaler.step(opt);scaler.update()
                total+=loss.item()*len(ids)
            history.append(total/len(order))
            if method=="mining":
                model.eval()
                with torch.no_grad(): hardness=np.concatenate([model(*batch(g))[0].sigmoid().cpu().numpy() for g in np.array_split(tr,100)])*4
        model.eval()
        with torch.no_grad():
            def predict(indices,noise=False,shuffle_quality=False):
                logits=[]; lat=[]
                for g in np.array_split(indices,int(np.ceil(len(indices)/128))):
                    x,f,b=batch(g)
                    if shuffle_quality: f[:,14:]=f.flip(0)[:,14:]
                    if noise: x=corrupt(x,c["hz"])
                    z,h,_=model(x,f,b); logits.extend(z.cpu().numpy());lat.extend(h.cpu().numpy())
                return np.array(logits),np.array(lat)
            start=time.perf_counter(); z,_=predict(va); ms=1000*(time.perf_counter()-start)/len(va)
            if teacher:
                zz,hh=predict(tr); full=np.zeros(len(y),"f"); emb=np.zeros((len(y),48),"f")
                # Calibrate teacher supervision on training examples only; fold 9 is reserved.
                ab=fit_calibration(zz,y[tr]);full[tr]=zz*ab["slope"]+ab["intercept"];emb[tr]=hh
                np.savez(folder/"targets.npz",logit=full,latent=emb)
            score={} if final else devscore(self.m.iloc[va],z)
            if not final and c.get("fusion"):
                score["quality_ap_drop"]=score["ap"]-devscore(self.m.iloc[va],predict(va,shuffle_quality=True)[0])["ap"]
            if not final and (name=="E3" or c.get("augment")):
                seed_everything(78); score["corrupt_ap"]=devscore(self.m.iloc[va],predict(va,True)[0])["ap"]
        np.savez(folder/"scores.npz",ids=va,logit=z)
        torch.save(model.state_dict(),folder/"model.pt")
        row=dict(name=name,config=c,params=sum(p.numel() for p in model.parameters()),ms=ms,train_n=len(tr),train_positive=int(y[tr].sum()),history=history,**score)
        save_json(folder/"done.json",row);print(name,score,flush=True)
        return row

def corrupt(x,hz):
    n=len(x); t=torch.arange(x.shape[-1],device=x.device)/hz
    phase=torch.rand(n,1,1,device=x.device)*6.28
    gain=1+.015*torch.randn_like(phase)
    x=x.roll(int(torch.randint(-3,4,())),dims=-1)*gain+.02*torch.sin(t*1.2+phase)+.006*torch.sin(t*2*np.pi*50+phase)
    noise=torch.randn_like(x)*.005
    x=x+F.avg_pool1d(noise,5,1,2)+.01*torch.randn_like(phase)
    # Continuous minor clock perturbation with endpoint clamping.
    loc=(t*hz*(1+.001*torch.sin(phase))).clamp(0,x.shape[-1]-1)
    idx=loc.long().expand_as(x); frac=loc-loc.long()
    x=x.gather(-1,idx)*(1-frac)+x.gather(-1,(idx+1).clamp_max(x.shape[-1]-1))*frac
    # Short hold on a locally quiet segment, avoiding steep QRS edges.
    length=max(3,int(.025*hz))
    delta=F.pad(x.diff().abs(),(1,0))
    energy=F.max_pool1d(delta,length,stride=length)
    where=energy.argmin(-1,keepdim=True)*length
    ix=where+torch.arange(length,device=x.device)
    x=x.scatter(-1,ix,x.gather(-1,where).expand_as(ix))
    return (x.clamp(-4,4)*512).round()/512
