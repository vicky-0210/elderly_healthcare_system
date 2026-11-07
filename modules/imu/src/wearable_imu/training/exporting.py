"""Refit selected models; verify compact runtime predictions against fitted estimators."""
import json
from pathlib import Path
import platform,time
import numpy as np
import pandas as pd
from scipy.special import logit
from wearable_imu.datasets.sisfall import RESULTS,COUNTS_TO_PHYSICAL,load_cache
from wearable_imu.preprocessing.ipmv import wave_coeffs
from wearable_imu.inference.predictor import IMUPredictor
from . import protocol as p
from .catalogue import available

EXPORT=Path("artifacts/imu/exports/sisfall_v1")


def pack_estimator(bundle,kind,path):
    model,scaler=bundle
    center=scaler.center_ if hasattr(scaler,"center_") else scaler.mean_
    pack=dict(kind=kind,center=center,scale=scaler.scale_)
    if kind=="svm":
        s=model.support_vectors_
        pack.update(support=s,support_norm=(s*s).sum(1),dual=model.dual_coef_.flatten(),bias=model.intercept_[0],gamma=model._gamma)
    else:
        trees=[]
        if kind=="xgb":
            config=json.loads(model.get_booster().save_raw(raw_format="json"))["learner"]
            bias=float(str(config["learner_model_param"]["base_score"]).strip("[]"))
            pack["bias"]=np.float32(logit(bias))
            for t in config["gradient_booster"]["model"]["trees"]:
                trees.append((t["left_children"],t["right_children"],t["split_indices"],np.asarray(t["split_conditions"],dtype="f")))
        else:
            for t in model.estimators_:
                t=t.tree_; y=t.value[:,0,:]; value=np.where(t.children_left<0,y[:,1]/y.sum(1),t.threshold)
                trees.append((t.children_left,t.children_right,t.feature,value))
        roots=[]; left=[]; right=[]; features=[]; values=[]; depths=[]
        for a,b,f,v in trees:
            base=len(left); roots.append(base); a,b=np.asarray(a),np.asarray(b)
            left.extend(np.where(a>=0,a+base,-1)); right.extend(np.where(b>=0,b+base,-1))
            features.extend(np.maximum(f,0)); values.extend(v)
            frontier=[0]; depth=0
            while frontier:
                frontier=[child for node in frontier if a[node]>=0 for child in (a[node],b[node])]
                if frontier: depth+=1
            depths.append(depth)
        pack.update(root=np.asarray(roots,dtype="i4"),left=np.asarray(left,dtype="i4"),right=np.asarray(right,dtype="i4"),
            feature=np.asarray(features,dtype="i4"),value=np.array(values,dtype="f" if kind=="xgb" else "d"),depth=max(depths))
    np.savez_compressed(path,**pack)


def runtime_meta(spec,d,thr,ab):
    waves=[w for w in ("db2","sym4") if any(w in k for k in d["names"])]
    return dict(format=1,experiment=spec.name,sample_rate_hz=spec.settings.get("fs",200),input_units=["g"]*3+["deg/s"]*3,
        channels=d["names"],width=int(d["width"]),hop=int(round(d["width"]*(1-spec.overlap))),
        preprocessing=spec.settings,waves=waves,wave_coefficients={w:[a.tolist() for a in wave_coeffs(w)] for w in waves},
        threshold=thr,calibration=ab,training_population="38 SisFall participants, simulated trials",
        score_semantics="Sigmoid is calibrated on trial maxima. Window values are evidence for alerts.")


def measure(func,n=200):
    for _ in range(8): func()
    costs=[]
    for _ in range(n):
        start=time.perf_counter_ns(); func(); costs.append((time.perf_counter_ns()-start)/1e6)
    return dict(p50_ms=float(np.median(costs)),p95_ms=float(np.percentile(costs,95)),calls=n)


def export_one(key,root,out,public,device="cpu"):
    import torch,joblib
    spec=available()[key]; d=p.make_data(spec); ids=np.arange(len(d["m"]))
    rows=pd.concat([pd.read_csv(root/key/f"fold{k}"/"test.csv") for k in range(5)])
    _,thr=p.operating_point(rows.y.to_numpy(),rows.logit.to_numpy(),rows.subject.to_numpy())
    ab=p.calibration(rows.y.to_numpy(),rows.logit.to_numpy(),rows.subject.to_numpy())
    meta=runtime_meta(spec,d,thr,ab)
    out=out/key; out.mkdir(parents=True,exist_ok=True)
    starts=np.random.default_rng(33).choice(d["starts"],256,replace=False)
    probe=d["x"][starts[:,None]+np.arange(d["width"])]
    if spec.model in ("rf","svm","xgb"):
        from .classical import make_features,fit,predict
        from wearable_imu.models.classical import descriptors
        f=make_features(d); model=fit(spec,d,f,ids,p.SEED)
        joblib.dump(model,out/"refit.joblib",compress=3)
        pack_estimator(model,spec.model,out/"classifier.npz")
        meta.update(engine="numpy",file="classifier.npz")
        expected=predict(model,descriptors(probe)); del f
    else:
        from .neural import setup,train_phase,save
        epochs=int(np.median([json.loads((root/key/f"fold{k}"/"done.json").read_text())["epochs"] for k in range(5)]))
        data=setup(d,ids,spec.scaler,device)
        teacher=None
        if key=="dm_distilled":
            from .classical import make_features
            from .neural import fit_teacher
            teacher=fit_teacher(spec,d,make_features(d),ids,p.SEED,device)
        net,hist,_=train_phase(spec,d,data,ids,p.SEED,epochs,teacher=teacher); net=net.cpu().eval()
        save(out/"refit.pt",net,data,d,spec,thr,ab); (out/"refit_history.json").write_text(json.dumps(hist))
        probe_normalized=torch.tensor(p.normalize(probe,data[1],data[2])).transpose(1,2).contiguous()
        with torch.no_grad(): expected=net(probe_normalized).numpy()
        torch.onnx.export(net,probe_normalized[:1],str(out/"model.onnx"),input_names=["window"],output_names=["logit"],
            opset_version=13,do_constant_folding=True)
        import onnx
        onnx.checker.check_model(str(out/"model.onnx"))
        meta.update(engine="onnx",file="model.onnx",center=data[1].tolist(),scale=data[2].tolist(),epochs=epochs)
    (out/"runtime.json").write_text(json.dumps(meta,indent=2))
    engine=IMUPredictor(out); actual=engine.infer(probe)
    error=float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,atol=2e-4,rtol=2e-4): raise ValueError(f"Export mismatch {key}: {error}")
    bench=measure(lambda:engine.infer(probe[:1]))
    # Representative continuous frontend + classifier wall time per decision step.
    _,raw=load_cache(); record=d["m"].iloc[0]; data=(raw[record.offset:record.offset+record.n]*COUNTS_TO_PHYSICAL)
    if engine.fs==100:
        from scipy.signal import butter,sosfilt,sosfilt_zi
        sos=butter(4,40,fs=200,output="sos"); data=sosfilt(sos,data,axis=0,zi=sosfilt_zi(sos)[:,:,None]*data[0])[0][1::2]
    block=int(engine.hop*engine.fs/20); pointer=0
    def step():
        nonlocal pointer
        ix=(np.arange(block)+pointer)%len(data)
        engine.push(data[ix],pointer/engine.fs); pointer+=block
    streaming=measure(step)
    checksum=__import__("hashlib").sha256((out/meta["file"]).read_bytes()).hexdigest()
    result=dict(experiment=key,engine=meta["engine"],file_bytes=(out/meta["file"]).stat().st_size,
        metadata_bytes=(out/"runtime.json").stat().st_size,refit_model_bytes=(out/("refit.pt" if meta["engine"]=="onnx" else "refit.joblib")).stat().st_size,
        tested_windows=len(probe),max_logit_error=error,threshold_disagreements=int(np.sum((actual>=thr)!=(expected>=thr))),
        model_sha256=checksum,host=platform.platform(),processor=platform.processor(),single_window=bench,full_hop=streaming,
        window_s=engine.width/20,hop_s=engine.hop/20,final_threshold=thr,calibration=ab)
    (public/f"export_{key}.json").write_text(json.dumps(result,indent=2))
    print(key,json.dumps(result),flush=True)
    return result
