"""Run frozen held-out checkpoints on deterministic, hypothetical sensor perturbations."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.signal import butter,sosfilt,sosfilt_zi
from wearable_imu.datasets.sisfall import load_cache,RESULTS,COUNTS_TO_PHYSICAL
from wearable_imu.training.protocol import RUN,calibrated
from wearable_imu.training.catalogue import available
from wearable_imu.preprocessing.ipmv import IPMV
from wearable_imu.evaluation.compare import metrics

CONDITIONS=["reference","static_rotation","gyro_nominal_gain","bias_scale_noise","bandwidth_clip","100hz_replay"]


def lowpass(x,hz,fs=200):
    sos=butter(4,hz,fs=fs,output="sos")
    return sosfilt(sos,x,axis=0,zi=sosfilt_zi(sos)[:,:,None]*x[0])[0]


def perturb(x,key,rng):
    if key=="static_rotation":
        r=Rotation.random(random_state=rng).as_matrix()
        return (x.reshape(-1,2,3)@r).reshape(-1,6)
    if key=="gyro_nominal_gain": return x*np.array([1]*3+[16.384/14.375]*3)
    if key=="bias_scale_noise":
        x=x*rng.uniform(.95,1.05,6)
        x+=rng.uniform(-1,1,6)*np.array([.08]*3+[4]*3)
        x+=rng.normal(size=x.shape)*[.015,.015,.015,1.,1.,1.]
        lsb=np.array([1/2048]*3+[1/16.4]*3)
        return np.round(x/lsb)*lsb
    if key=="bandwidth_clip":
        return np.clip(lowpass(x,20),[-8]*3+[-1000]*3,[8]*3+[1000]*3)
    return x


def scorer(spec,folder):
    if spec.model in ("xgb","rf","svm"):
        import joblib
        from wearable_imu.training.classical import predict
        from wearable_imu.models.classical import descriptors
        bundle=joblib.load(folder/"model.joblib")
        return lambda w:predict(bundle,descriptors(w))
    import torch
    from wearable_imu.models.inertial import FallNet
    data=torch.load(folder/"model.pt",weights_only=True,map_location="cpu")
    net=FallNet(spec.model,data["names"],**spec.network).eval()
    net.load_state_dict(data["state"])
    def forward(w):
        with torch.inference_mode():
            x=torch.asinh((torch.as_tensor(w)-data["center"])/data["scale"]).transpose(1,2)
            return torch.cat([net(b) for b in x.split(256)]).numpy()
    return forward


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--experiments",nargs="+"); args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    choices=json.loads((RESULTS/"selection.json").read_text())
    keys=args.experiments or list(dict.fromkeys(choices[g]["final"] for g in ("all","neural")))
    m,raw=load_cache(); output=[]
    for key in keys:
        spec=available()[key]
        for fold in range(5):
            folder=RUN/key/f"fold{fold}"; meta=json.loads((folder/"done.json").read_text())
            fn=scorer(spec,folder)
            test=m[m.fold==fold]
            ref=pd.read_csv(folder/"test.csv").set_index("id")
            for condition in CONDITIONS:
                records=[]
                config={**spec.settings}
                fs=100 if condition=="100hz_replay" else config.get("fs",200)
                config["fs"]=fs
                waves=[w for w in ("db2","sym4") if any(w in k for k in meta["names"])]
                ids=[IPMV(**config,waves=waves).names.index(k) for k in meta["names"]]
                width=meta["width"]; hop=int(round(width*(1-spec.overlap)))
                for block in np.array_split(test.index,max(1,len(test)//32)):
                    windows=[]; counts=[]
                    for i in block:
                        r=m.loc[i]; x=raw[r.offset:r.offset+r.n]*COUNTS_TO_PHYSICAL
                        data=perturb(x,condition,np.random.default_rng(987+i))
                        if fs==100: data=lowpass(data,40)[1::2]
                        ts=IPMV(**config,waves=waves).push(data)[:,ids]
                        starts=np.arange(0,len(ts)-width+1,hop)
                        windows.append(ts[starts[:,None]+np.arange(width)]); counts.append(len(starts))
                    logits=fn(np.concatenate(windows))
                    maxima=np.maximum.reduceat(logits,np.r_[0,np.cumsum(counts)[:-1]])
                    frame=m.loc[block,["id","subject","older","y","activity","fold"]].copy()
                    frame["p_fall"]=calibrated(maxima,meta["calibration"])
                    frame["decision"]=(maxima>=meta["threshold"]).astype(int)
                    frame["reference_decision"]=ref.loc[frame.id,"decision"].to_numpy()
                    frame["logit_diff"]=maxima-ref.loc[frame.id,"logit"].to_numpy()
                    frame["model"]=key; frame["condition"]=condition
                    records.append(frame)
                output.extend(records)
            print("Transfer replay",key,fold,flush=True)
    allrows=pd.concat(output)
    suffix="_distilled" if keys==["dm_distilled"] else ("_secondary" if args.experiments else "")
    allrows.to_csv(RESULTS/f"transfer_trials{suffix}.csv.gz",compression="gzip",index=False)
    summary=[dict(model=k,condition=c,**metrics(r),decision_changes=int((r.decision!=r.reference_decision).sum()),
        max_abs_logit_change=r.logit_diff.abs().max()) for (k,c),r in allrows.groupby(["model","condition"])]
    frame=pd.DataFrame(summary); frame.to_csv(RESULTS/f"transfer{suffix}.csv",index=False)
    from wearable_imu.evaluation.figures import plt,plotfile
    ax=frame.pivot(index="condition",columns="model",values="subject_ba").plot.barh(figsize=(9,4))
    ax.set(xlabel="Held-out subject-balanced accuracy",ylabel="",title="Frozen-model replay under simulated sensor differences"); plotfile(ax.figure,RESULTS/"figures"/f"transfer{suffix}.png")
    print(frame[["model","condition","subject_ba","decision_changes"]].to_string(index=False))

if __name__=="__main__": main()
