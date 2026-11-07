"""Measure FP16 sensitivity on actual held-out predictions, then export a half model."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import torch
from wearable_imu.training import protocol as p
from wearable_imu.training.catalogue import available
from wearable_imu.training.exporting import EXPORT
from wearable_imu.models.inertial import FallNet
from wearable_imu.datasets.sisfall import RESULTS
from wearable_imu.evaluation.compare import metrics


def load_net(path,spec,device):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    net=FallNet(spec.model,ck["names"],**spec.network)
    net.load_state_dict(ck["state"])
    return net.eval().to(device).half(),ck


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--model"); args=parser.parse_args()
    torch.set_num_threads(2); device="cuda" if torch.cuda.is_available() else "cpu"
    key=args.model or json.loads((RESULTS/"selection.json").read_text())["neural"]["final"]
    spec=available()[key]; d=p.make_data(spec); measures=[]
    for fold in range(5):
        root=p.RUN/key/f"fold{fold}"; net,ck=load_net(root/"model.pt",spec,device)
        r=pd.read_csv(root/"test.csv")
        starts=d["starts"][np.isin(d["group"],np.flatnonzero(d["folds"]==fold))]
        scores=[]
        with torch.inference_mode():
            for sl in np.array_split(starts,max(1,int(np.ceil(len(starts)/256)))):
                x=torch.from_numpy(d["x"][sl[:,None]+np.arange(d["width"])])
                x=torch.asinh((x-ck["center"])/ck["scale"]).transpose(1,2).to(device,dtype=torch.float16)
                scores.extend(net(x).float().cpu().numpy())
        z=p.trial_max(d,np.flatnonzero(d["folds"]==fold),np.asarray(scores))
        r["baseline_decision"]=r.decision; r["error"]=np.abs(r.logit-z)
        r["decision"]=(z>=ck["threshold"]).astype(int); r["p_fall"]=p.calibrated(z,ck["calibration"])
        measures.append(r)
    data=pd.concat(measures); result=metrics(data)
    root=EXPORT/key; net,ck=load_net(root/"refit.pt",spec,device)
    x=torch.zeros((1,len(ck["names"]),ck["width"]),dtype=torch.float16,device=device)
    torch.onnx.export(net,x,str(root/"model_fp16.onnx"),input_names=["window"],output_names=["logit"],opset_version=13)
    import onnx
    onnx.checker.check_model(str(root/"model_fp16.onnx"))
    result.update(experiment=key,backend=f"PyTorch {device} FP16",decision_changes=int((data.decision!=data.baseline_decision).sum()),
        max_logit_error=float(data.error.max()),fp16_bytes=(root/"model_fp16.onnx").stat().st_size,
        fp32_bytes=(root/"model.onnx").stat().st_size)
    (RESULTS/f"precision_{key}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
