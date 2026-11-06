"""Fixed-shape ONNX and self-contained research bundle; desktop parity check."""
from pathlib import Path
import copy,json,shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wearable_ecg.models.backbones import ECGClassifier,DiagonalSSM
from wearable_ecg.runtime import save_json,digest


class FrozenSSM(nn.Module):
    """Replace a fixed-length FFT convolution by an equivalent grouped Conv1d."""
    def __init__(self,block,length=300):
        super().__init__(); self.norm=block.norm; self.ff=block.ff
        self.register_buffer("kernel",block.kernel(length).detach().flip(-1)[:,None])
        self.register_buffer("skip",block.skip.detach()[None,:,None]); self.length=length
    def forward(self,x):
        z=self.norm(x)
        y=F.conv1d(F.pad(z,(self.length-1,0)),self.kernel,groups=z.shape[1])+self.skip*z
        return x+self.ff(F.gelu(y))


def export_bundle(run,cache,output):
    import onnx
    import onnxruntime as ort
    run,cache,output=map(Path,(run,cache,output)); output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(run/"refit/model.pt",map_location="cpu",weights_only=True)
    model=ECGClassifier(**checkpoint["model_config"]); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    export_model=copy.deepcopy(model)
    for i,block in enumerate(export_model.blocks):
        if isinstance(block,DiagonalSSM): export_model.blocks[i]=FrozenSSM(block)
    config=json.loads((run/"config.json").read_text())
    waves=np.load(cache/f"{config['preprocessing']}_waves.npy",mmap_mode="r")
    features=np.load(cache/f"{config['preprocessing']}_features.npy",mmap_mode="r")
    # Training records for serialization/parity checks, never a model-selection test.
    import pandas as pd
    manifest=pd.read_csv(cache/"manifest.csv")
    ids=manifest.query("strat_fold <= 6").row_index.to_numpy()[:8]
    x=torch.from_numpy(np.array(waves[ids[:1]])); f=torch.from_numpy(np.array(features[ids[:1]]))
    path=output/"model.onnx"
    with torch.inference_mode():
        torch.onnx.export(export_model,(x,f),str(path),input_names=["waveform_mv","voltage_features"],output_names=["lvh_logit"],
                          opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(path),sess_options=opts,providers=["CPUExecutionProvider"])
    errors=[]; frozen_errors=[]
    with torch.inference_mode():
        for idx in ids:
            xx=np.array(waves[idx:idx+1]); ff=np.array(features[idx:idx+1])
            reference=model(torch.from_numpy(xx),torch.from_numpy(ff)).numpy()
            frozen=export_model(torch.from_numpy(xx),torch.from_numpy(ff)).numpy()
            result=session.run(None,{"waveform_mv":xx,"voltage_features":ff})[0]
            errors.append(float(np.max(abs(result-reference)))); frozen_errors.append(float(np.max(abs(frozen-reference))))
    if max(errors)>1e-3: raise ValueError(f"ONNX parity failed: {max(errors)}")
    shutil.copy2(run/"refit/model.pt",output/"model.pt")
    shutil.copy2(run/"calibration.json",output/"calibration.json")
    shutil.copy2(cache/"preprocessing/controller.joblib",output/"controller.joblib")
    save_json(output/"inference_contract.json",{"lead":"II","units":"mV","sampling_rate_hz":360,"samples":3600,"window_seconds":10,
        "proposed_hop_seconds":2,"buffered_acausal_preprocessing":True,"preprocessing":config["preprocessing"],"task":config["task"],
        "architecture":config["architecture"],"seed":config["seed"],"onnx_sha256":digest(path),
        "preprocessing_precision":"Preserve float64 physical calibration through resampling; canonical waveform/features and ONNX inputs are float32",
        "hardware_domain_validated":False,"jetson_tested":False,"precision":"FP32; FP16 TensorRT requires device validation"})
    save_json(output/"export_parity.json",{"training_windows_checked":len(ids),"max_absolute_logit_error":max(errors),
        "max_frozen_ssm_logit_error":max(frozen_errors),"tolerance":.001,"provider":"ONNX Runtime CPU","passed":True})
    return output
