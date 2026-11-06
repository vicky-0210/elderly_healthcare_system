"""Subject-grouped development, 14-subject refit, and frozen outer predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
import torch
from torch.nn import functional as F

from wearable_eda.models.wrist import WristNet
from wearable_eda.inference.neural import fit_scaler, scale
from wearable_eda.training.neural_experiments import arguments, catalogue, selected_channels
from wearable_eda.training.wrist_ssl import pretrain

# Equal, predeclared, compact search budget for all model families.
MAX_EPOCHS, PATIENCE, BATCH, WIDTH, SEED = 6, 3, 256, 16, 971
VERSION = "wrist-neural-v1"


def dumps(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False),encoding="utf-8")


def pick_subjects(groups, outer):
    ids = np.unique(groups)
    if len(ids) < 4 or outer not in ids:
        raise ValueError("Need at least four distinct subjects and a valid holdout")
    index = list(ids).index(outer)
    inner = np.unique(ids[[(index+1)%len(ids),(index+len(ids)//2+1)%len(ids)]])
    return ids[~np.isin(ids,np.r_[outer,inner])], inner, ids[ids!=outer]


def load_windows(cache, options):
    from wearable_eda.datasets.wesad import build_windows
    d = build_windows(cache, seconds=60, stride=15, purity=.8, **options)
    if not np.isfinite(d["X"]).all() or not np.isfinite(d["F"]).all():
        raise ValueError("Preprocessing returned non-finite inputs")
    return d


def raw_view(d, names):
    if names == list(d["channels"]):
        return d["X"]
    return d["X"][:,[list(d["channels"]).index(n) for n in names]]


def prepare(d, raw, ids, names, device, ssl=None):
    train = np.isin(d["groups"],ids)
    scaler = fit_scaler(raw,d["F"],train,names)
    x,f = scale(raw,d["F"],scaler)
    pack = dict(x=torch.as_tensor(x,device=device),f=torch.as_tensor(f,device=device),
                y=torch.as_tensor(d["y"],dtype=torch.long,device=device),scaler=scaler,
                train=torch.as_tensor(np.flatnonzero(train),device=device))
    if ssl is not None:
        use = np.isin(ssl["groups"],ids)
        X,F = scale(raw_view(ssl,names)[use],ssl["F"][use],scaler)
        pack["ssl"] = tuple(torch.as_tensor(a,device=device) for a in
                            (X,F,ssl["groups"][use],ssl["start"][use]))
    return pack


def macro_subject(probs, y, groups):
    pred = probs.argmax(1)
    scores = []
    for subject in np.unique(groups):
        idx = groups==subject
        confusion = np.bincount(y[idx]*probs.shape[1]+pred[idx],minlength=probs.shape[1]**2).reshape(probs.shape[1],-1)
        den = confusion.sum(0)+confusion.sum(1)
        scores.append(np.divide(2*np.diag(confusion),den,out=np.zeros(len(den)),where=den>0).mean())
    return float(np.mean(scores))


def batch_logits(model, pack, idx):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(pack["x"][part],pack["f"][part])[0] for part in idx.split(BATCH)]).cpu().numpy()


def temperature(logits, y):
    from scipy.special import logsumexp
    def nll(log_t):
        z = logits/np.exp(log_t)
        return (logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean()
    fit = scipy.optimize.minimize_scalar(nll, bounds=(-1.6,1.6), method="bounded")
    return float(np.exp(fit.x))


def probabilities(logits, temp):
    from scipy.special import softmax
    return softmax(logits.astype(np.float64)/temp,axis=1)


def train_phase(spec, names, features, pack, seed, epochs, validation=None):
    device = pack["x"].device
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = WristNet(spec,names,len(features),len(spec["classes"]),
                     pack["scaler"]["xc"],pack["scaler"]["xs"]).to(device)
    train = pack["train"]
    history = []
    if spec.get("ssl"):
        a,b,g,t = pack["ssl"]
        history = pretrain(model,a,b,g,t,torch.arange(len(a),device=device),spec["ssl"],generator)
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=.0002)
                 if spec.get("schedule") == "cosine" else None)
    counts = torch.bincount(pack["y"][train],minlength=len(spec["classes"]))
    if (counts==0).any():
        raise ValueError("Missing class on training subjects")
    weight = len(train)/(len(counts)*counts.float())
    best, best_epoch, best_logits = (-1.,-float("inf")), 0, None
    for epoch in range(1,epochs+1):
        model.train()
        order = train[torch.randperm(len(train),device=device,generator=generator)]
        loss_sum, pen_sum = 0.,0.
        for idx in order.split(BATCH):
            logits, penalty = model(pack["x"][idx],pack["f"][idx])
            y = pack["y"][idx]
            ce = F.cross_entropy(logits,y,weight=weight,reduction="none")
            if spec.get("loss") == "focal":
                ce = ce*(1-logits.softmax(1).gather(1,y[:,None]).flatten()).square()
            loss = ce.mean()+spec.get("physics_weight",.1)*penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            loss_sum += loss.detach()*len(idx)
            pen_sum += penalty.detach()*len(idx)
        row = dict(epoch=epoch,phase="supervised",loss=float(loss_sum/len(train)),
                   decomposition_loss=float(pen_sum/len(train)))
        if not np.isfinite(row["loss"]):
            raise FloatingPointError("Nonfinite training objective")
        if validation is not None:
            idx, y, groups = validation
            logits = batch_logits(model,pack,idx)
            probs = probabilities(logits,1.)
            metric = macro_subject(probs,y,groups)
            nll = -np.log(probs[np.arange(len(y)),y].clip(1e-9)).mean()
            row.update(val_macro_f1=metric,val_nll=float(nll))
            if (metric,-nll) > best:
                best, best_epoch, best_logits = (metric,-nll),epoch,logits
        row["lr"] = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        history.append(row)
        if validation is not None and epoch-best_epoch>=PATIENCE:
            break
    return model, history, best_epoch, best_logits, best[0]


def save_checkpoint(path, spec, names, features, model, pack, temp):
    torch.save(dict(format=VERSION,spec=spec,channels=names,features=list(features),temperature=temp,
                    scaler={k:torch.from_numpy(v.copy()) for k,v in pack["scaler"].items()},
                    weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    window_samples=pack["x"].shape[-1]), path)


def pred_frame(d, indices, logits, temp):
    probs = probabilities(logits,temp)
    frame = pd.DataFrame(dict(subject=d["groups"][indices],start=d["start"][indices],
                              end=d["end"][indices],y_true=d["y"][indices]))
    for c in range(probs.shape[1]):
        frame[f"p{c}"] = probs[:,c]
    if "motion" in d["channels"]:
        frame["motion"] = d["X"][indices,list(d["channels"]).index("motion")].mean(1)
    return frame


def fold(spec, path, d, names, dev, refit, train_ids, val_ids, outer, signature):
    tic = time.perf_counter()
    test_ids = np.flatnonzero(d["groups"]==outer)
    val = np.flatnonzero(np.isin(d["groups"],val_ids))
    device = dev["x"].device
    seed = SEED+int(outer)
    _, hist, ep, logits, score = train_phase(spec,names,d["features"],dev,seed,spec.get("epoch_limit", MAX_EPOCHS),
        (torch.as_tensor(val,device=device),d["y"][val],d["groups"][val]))
    temp = temperature(logits,d["y"][val])
    path.mkdir(parents=True,exist_ok=True)
    pred_frame(d,val,logits,temp).to_csv(path/"inner_predictions.csv",index=False)
    model, fit_hist, *_ = train_phase(spec,names,d["features"],refit,seed,ep)
    test_logits = batch_logits(model,refit,torch.as_tensor(test_ids,device=device))
    pred_frame(d,test_ids,test_logits,temp).to_csv(path/"predictions.csv",index=False)
    save_checkpoint(path/"model.pt",spec,names,d["features"],model,refit,temp)
    pd.DataFrame([dict(stage=s,**r) for s,rows in (("inner",hist),("refit",fit_hist)) for r in rows]).to_csv(path/"history.csv",index=False)
    rec = dict(subject=int(outer),inner_score=float(score),parameters=json.dumps(dict(epochs=ep,width=WIDTH,
               lr=.002,batch=BATCH,temperature=temp)),epochs=ep,temperature=temp,seed=seed,
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               artifact_bytes=(path/"model.pt").stat().st_size,inner_train_subjects=list(map(int,train_ids)),
               inner_validation_subjects=list(map(int,val_ids)),refit_subjects=list(map(int,np.r_[train_ids,val_ids])),
               wall_seconds=round(time.perf_counter()-tic,3),signature=signature)
    dumps(path/"fold.tmp",rec)
    (path/"fold.tmp").replace(path/"fold.json")


def aggregate(folder, subjects):
    dirs = [folder/f"S{int(s):02d}" for s in subjects]
    if all((p/"fold.json").exists() for p in dirs):
        pd.concat([pd.read_csv(p/"predictions.csv") for p in dirs],ignore_index=True).to_csv(folder/"predictions.tmp",index=False)
        (folder/"predictions.tmp").replace(folder/"predictions.csv")
        pd.DataFrame([json.loads((p/"fold.json").read_text()) for p in dirs]).to_csv(folder/"folds.tmp",index=False)
        (folder/"folds.tmp").replace(folder/"folds.csv")


def config_sig(spec, cache):
    manifest = (Path(cache)/"manifest.json").read_bytes()
    return hashlib.sha256(manifest+json.dumps([VERSION,spec,SEED,MAX_EPOCHS,BATCH,PATIENCE],sort_keys=True).encode()).hexdigest()


def run_suite(cache, run, names, device):
    cache, run = Path(cache),Path(run)
    cat = catalogue()
    batches = defaultdict(list)
    for name in names:
        spec = cat[name]
        batches[json.dumps([arguments(spec),spec.get("drop")],sort_keys=True)].append(spec)
    for key, specs in batches.items():
        options,_ = json.loads(key)
        d = load_windows(cache,options)
        subjects = np.unique(d["groups"])
        if len(subjects)!=15:
            raise ValueError(f"Expected 15 WESAD subjects, found {subjects}")
        channels = selected_channels(d["channels"],specs[0])
        raw = raw_view(d,channels)
        ssl = load_windows(cache,dict(options,classes=[1,2,3,4])) if any(s.get("ssl") for s in specs) else None
        signatures = {}
        for spec in specs:
            folder = run/"neural"/spec["id"]
            folder.mkdir(parents=True,exist_ok=True)
            signatures[spec["id"]] = config_sig(spec,cache)
            dumps(folder/"meta.json",dict(**spec,seconds=60,stride=15,purity=.8,channels=channels,
                  features=list(d["features"]),calibrator="inner scalar temperature",width=WIDTH,
                  max_epochs=spec.get("epoch_limit",MAX_EPOCHS),batch_size=BATCH,patience=PATIENCE,
                  seed=SEED,device=str(device),torch_version=str(torch.__version__),
                  ssl_eligibility="Training subjects only, four-state eligible windows; state identities discarded"))
        for outer in subjects:
            tic = time.perf_counter()
            todo = []
            for spec in specs:
                path = run/"neural"/spec["id"]/f"S{int(outer):02d}"
                sig = signatures[spec["id"]]
                if (path/"fold.json").exists():
                    if json.loads((path/"fold.json").read_text())["signature"] != sig:
                        raise ValueError(f"Changed protocol for existing {path}; use a new run directory")
                else:
                    todo.append((spec,path,sig))
            if not todo:
                continue
            a,v,r = pick_subjects(d["groups"],outer)
            dev = prepare(d,raw,a,channels,device,ssl)
            refit = prepare(d,raw,r,channels,device,ssl)
            for spec,path,sig in todo:
                fold(spec,path,d,channels,dev,refit,a,v,outer,sig)
            del dev,refit
            print(f"{specs[0]['id']} group: S{outer}, {len(todo)} models in {time.perf_counter()-tic:.1f}s",flush=True)
        for spec in specs:
            aggregate(run/"neural"/spec["id"],subjects)


def export_full(cache,run,experiment,destination,device="cpu"):
    """Train independent deployable model using pooled inner choices only."""
    torch.set_num_threads(2)
    spec = catalogue()[experiment]
    folder = Path(run)/"neural"/experiment
    folds = pd.read_csv(folder/"folds.csv")
    if len(folds)!=15 or folds.subject.nunique()!=15:
        raise ValueError("All 15 grouped inner choices required")
    d = load_windows(cache,arguments(spec))
    names = selected_channels(d["channels"],spec)
    ssl = load_windows(cache,dict(arguments(spec),classes=[1,2,3,4])) if spec.get("ssl") else None
    pack = prepare(d,raw_view(d,names),np.unique(d["groups"]),names,device,ssl)
    ep = int(folds.epochs.median())
    model,*_ = train_phase(spec,names,d["features"],pack,SEED,ep)
    path = Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = float(folds.temperature.median())
    save_checkpoint(path,spec,names,d["features"],model,pack,temp)
    return dict(path=str(path),epochs=ep,temperature=temp,subjects=np.unique(d["groups"]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    parser.add_argument("--run",type=Path,default=Path("artifacts/eda/runs/wesad_loso_v1"))
    parser.add_argument("--models",nargs="+",default=["all"])
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.list:
        print("\n".join(catalogue()))
        return
    torch.backends.cudnn.benchmark = False
    run_suite(args.cache,args.run,list(catalogue()) if args.models==["all"] else args.models,args.device)

