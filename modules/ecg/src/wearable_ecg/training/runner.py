"""Patient-independent development, refitting, and fold-9 calibration.

No test predictions or metrics are computed here. Evaluation is a separate
locked stage after every model's configuration and threshold have been saved.
"""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from wearable_ecg.datasets.prepare import load_cache
from wearable_ecg.labels.ptbxl import cohort
from wearable_ecg.models.backbones import make_model
from wearable_ecg.evaluation.metrics import metrics, choose_threshold, choose_sensitive_threshold, fit_calibration, calibrated
from wearable_ecg.runtime import save_json, seed_everything, environment, digest


@torch.inference_mode()
def predict_logits(model,x,features,batch=128):
    model.eval(); chunks=[]
    for start in range(0,len(x),batch):
        # Freeze calibration/evaluation in FP32 to match the research export.
        with torch.autocast(device_type=x.device.type,enabled=False):
            chunks.append(model(x[start:start+batch],features[start:start+batch]).float().cpu().numpy())
    return np.concatenate(chunks)


def get_optimizer(model,lr,weight_decay):
    # Stable SSM poles/dt: no weight decay, smaller learning rate.
    special=[p for n,p in model.named_parameters() if "log_decay" in n or "log_dt" in n]
    ids={id(p) for p in special}
    regular=[p for p in model.parameters() if id(p) not in ids]
    groups=[{"params":regular,"lr":lr,"weight_decay":weight_decay}]
    if special: groups.append({"params":special,"lr":lr*.1,"weight_decay":0.})
    return torch.optim.AdamW(groups)


def fit(model,x,features,y,cfg,output,dev=None,fixed_epochs=None,kd=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    device=x.device; model.to(device)
    optimizer=get_optimizer(model,cfg["learning_rate"],cfg["weight_decay"])
    count=int(y.sum().item()); pos_weight=torch.tensor([(len(y)-count)/max(count,1)],device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    epochs=fixed_epochs or cfg["max_epochs"]
    # Same fixed LR in development/refit avoids horizon-dependent schedule drift.
    history=[]; best_score=(-1.,-1.); best_epoch=0; stall=0
    t=torch.arange(x.shape[-1],device=device)/360
    start_all=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); order=torch.randperm(len(x),device=device); total=0.; begin=time.perf_counter()
        for ids in order.split(cfg["batch_size"]):
            xb=x[ids]; fb=features[ids]; yb=y[ids]
            if cfg.get("augmentation",True):
                # Mild additive acquisition noise only; no gain/time/polarity edits.
                noise=torch.randn_like(xb)*.003
                phase=torch.rand(len(ids),device=device)*2*np.pi
                drift=.008*torch.sin(2*np.pi*.25*t[None]+phase[:,None])
                xb=xb+noise+(drift if xb.ndim==2 else drift[:,None])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(xb,fb)
                loss=criterion(logits,yb)
                if kd is not None:
                    target,gate=kd
                    T=2.0
                    soft=target[ids]
                    student=logits.float()/T
                    # Bernoulli KL up to teacher-only entropy, weighted per record.
                    kl=nn.functional.binary_cross_entropy_with_logits(student,soft,reduction="none")
                    loss=loss+.5*T*T*(gate[ids]*kl).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            total+=float(loss.detach())*len(ids)
        row={"epoch":epoch,"train_loss":total/len(x),"seconds":time.perf_counter()-begin}
        if dev is not None:
            vx,vf,vy=dev; logits=predict_logits(model,vx,vf)
            p=expit(logits); threshold=choose_threshold(vy,p)
            m=metrics(vy,p,threshold)
            row.update({"dev_auroc":m["auroc"],"dev_auprc":m["auprc"],
                        "dev_sensitivity_at_90spec":m["sensitivity"],"dev_specificity":m["specificity"],
                        "dev_bce":float(np.mean(np.logaddexp(0,logits)-vy*logits))})
            score=(round(m["sensitivity"],5),m["auprc"])
            if score>best_score:
                best_score=score; best_epoch=epoch; stall=0
                torch.save({"model_config":model.config,"state_dict":model.state_dict(),"epoch":epoch},output/"best.pt")
            else: stall+=1
        else: best_epoch=epoch
        history.append(row)
        pd.DataFrame(history).to_csv(output/"history.csv",index=False)
        print(json.dumps({"run":output.parent.name,"phase":output.name,**row}),flush=True)
        if dev is not None and stall>=cfg["patience"]: break
    elapsed=time.perf_counter()-start_all
    if dev is None:
        torch.save({"model_config":model.config,"state_dict":model.state_dict(),"epoch":best_epoch},output/"model.pt")
    return {"best_epoch":best_epoch,"epochs_ran":len(history),"training_seconds":elapsed,
            "parameters":sum(p.numel() for p in model.parameters()),"class_weight":float(pos_weight.item())}


def tensors(waves,features,frame,device):
    ids=frame.row_index.to_numpy()
    return (torch.tensor(np.array(waves[ids]),device=device,dtype=torch.float32),
            torch.tensor(np.array(features[ids]),device=device,dtype=torch.float32),
            torch.tensor(frame.label.to_numpy(),device=device,dtype=torch.float32))


def train_run(cache,root,task,architecture,seed,cfg,variant="constrained",options=None):
    options=options or {}
    run=Path(root)/task/f"{architecture}__{variant}__seed{seed}"
    if (run/"COMPLETE.json").exists():
        previous=json.loads((run/"config.json").read_text())
        if previous["training"]!=cfg or previous["model_options"]!=options:
            raise ValueError("Completed run configuration differs; choose a new output directory")
        print("Reusing completed training",run,flush=True); return run
    seed_everything(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame,waves,features=load_cache(cache,variant)
    selected=cohort(frame,task)
    inner=selected.query("strat_fold <= 6"); dev=selected.query("strat_fold in [7,8]")
    final=selected.query("strat_fold <= 8"); cal=selected.query("strat_fold == 9")
    if options.get("input_channels",1)==12:
        waves=np.load(Path(cache)/"teacher_12lead_360hz_mv.npy",mmap_mode="r")
    run.mkdir(parents=True,exist_ok=True)
    frozen={"task":task,"architecture":architecture,"seed":seed,"preprocessing":variant,
            "training":cfg,"model_options":options,"development_folds":[7,8],"refit_folds":list(range(1,9)),
            "calibration_fold":9,"test_fold":10,"selection":"sensitivity at development 90% specificity, then average precision",
            "source_metadata_sha256":json.loads((Path(cache)/"COMPLETE.json").read_text())["metadata_sha256"]}
    save_json(run/"config.json",frozen)
    tx,tf,ty=tensors(waves,features,inner,device)
    vx,vf,vy=tensors(waves,features,dev,device)
    model=make_model(architecture,tf.cpu(),**options)
    selection=fit(model,tx,tf,ty,cfg,run/"development",dev=(vx,vf,vy.cpu().numpy()))
    del tx,tf,ty,vx,vf,vy,model
    gc.collect(); torch.cuda.empty_cache()
    seed_everything(seed)
    tx,tf,ty=tensors(waves,features,final,device)
    model=make_model(architecture,tf.cpu(),**options)
    refit=fit(model,tx,tf,ty,cfg,run/"refit",fixed_epochs=selection["best_epoch"])
    del tx,tf,ty; gc.collect(); torch.cuda.empty_cache()
    cx,cf,cy=tensors(waves,features,cal,device)
    logits=predict_logits(model,cx,cf)
    calibrator=fit_calibration(logits,cy.cpu().numpy())
    probs=calibrated(logits,calibrator)
    threshold=choose_threshold(cal.label.to_numpy(),probs,cfg["target_specificity"])
    high_sensitivity=choose_sensitive_threshold(cal.label.to_numpy(),probs,.95)
    output=cal[["ecg_id","patient_id","label","strat_fold"]].copy()
    output["logit"]=logits; output["probability"]=probs
    output.to_csv(run/"calibration_predictions.csv",index=False)
    save_json(run/"calibration.json",{**calibrator,"threshold":threshold,"high_sensitivity_threshold":high_sensitivity,
        "target_specificity":cfg["target_specificity"],"calibration_metrics":metrics(cal.label,probs,threshold),
        "note":"Fold-9 fitted calibration/operating point; not unbiased validation performance."})
    save_json(run/"COMPLETE.json",{"development":selection,"refit":refit,"model_sha256":digest(run/"refit/model.pt"),
                                   "test_evaluated":False})
    del model,cx,cf,cy; gc.collect(); torch.cuda.empty_cache()
    return run


def train_suite(cache,root,config):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/"TEST_EVALUATION_STARTED.json").exists():
        raise RuntimeError("Test has been opened for this experiment. Start a new, explicitly exploratory experiment to change models.")
    save_json(root/"environment.json",environment())
    save_json(root/"experiment.json",config)
    runs=[]
    for task in config["tasks"]:
        for architecture in config["architectures"]:
            for seed in config["seeds"]:
                runs.append(str(train_run(cache,root,task,architecture,seed,config["training"])))
    save_json(root/"FROZEN_FOR_TEST.json",{"runs":runs,"configuration_sha256":digest(root/"experiment.json"),
        "rule":"All configurations, epochs, calibration and thresholds fixed before fold-10 scoring."})
    print("All training complete; final comparison frozen",flush=True)
