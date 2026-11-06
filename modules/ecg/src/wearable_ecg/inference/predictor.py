"""Buffered Lead II research inference using the exact training preprocessing."""
from pathlib import Path
import json
import numpy as np
import joblib
import torch
from wearable_common.inference import SignalPredictor
from wearable_ecg.models.backbones import ECGClassifier
from wearable_ecg.preprocessing.pipeline import process,resample_mv
from wearable_ecg.evaluation.metrics import calibrated


class ECGPredictor(SignalPredictor):
    """Single 10-second calibrated window; no patient demographics are inputs."""
    def __init__(self): self.loaded=False

    def load(self,artifact_directory):
        root=Path(artifact_directory)
        self.contract=json.loads((root/"inference_contract.json").read_text())
        checkpoint=torch.load(root/"model.pt",map_location="cpu",weights_only=True)
        self.model=ECGClassifier(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["state_dict"]); self.model.eval()
        self.calibrator=json.loads((root/"calibration.json").read_text())
        # Load only a locally produced/trusted model bundle; joblib is executable.
        self.controller=joblib.load(root/"controller.joblib")
        self.loaded=True
        return self

    def predict(self,window):
        if not self.loaded: raise RuntimeError("Load a trusted ECG model bundle first")
        if not isinstance(window,dict): raise ValueError("Window must include samples, sampling_rate_hz, lead and units")
        if window.get("lead")!="II" or window.get("units")!="mV":
            return {"status":"insufficient_signal","reason":"Verified Lead II placement and mV calibration required","score":None}
        rate=window.get("sampling_rate_hz")
        if rate not in (360,500): raise ValueError("Supported validated offline rates: 360 or 500 Hz")
        x=np.asarray(window["samples"],dtype=np.float64)
        if x.ndim!=1 or len(x)!=10*rate or not np.isfinite(x).all():
            return {"status":"insufficient_signal","reason":"Require a complete finite 10-second window","score":None}
        if "timestamps" in window:
            times=np.asarray(window["timestamps"],dtype=float)
            if times.shape!=x.shape or not np.isfinite(times).all() or np.any(np.abs(np.diff(times)-1/rate)>.2/rate):
                return {"status":"insufficient_signal","reason":"Timestamp discontinuity or unverified sample clock","score":None}
        if np.std(x)<1e-5 or np.any(window.get("adc_clipped",False)):
            return {"status":"insufficient_signal","reason":"Flat or ADC-clipped signal","score":None}
        if rate!=360: x=resample_mv(x,rate,360)
        y,features,quality,setting=process(x,self.controller,self.contract["preprocessing"])
        if float(np.mean(quality))<.5:
            return {"status":"insufficient_signal","reason":"Signal integrity checks failed","score":None}
        with torch.inference_mode():
            logit=self.model(torch.from_numpy(y[None]),torch.from_numpy(features[None])).item()
        probability=float(calibrated([logit],self.calibrator)[0])
        return {"status":"research_score","target":"LVH-associated ECG diagnostic statement",
                "score":probability,"flag":probability>=self.calibrator["threshold"],
                "threshold":self.calibrator["threshold"],"quality":float(np.mean(quality)),
                "preprocessing_setting":setting,"window_seconds":10,
                "hardware_domain_validated":False,"anatomical_lvh_diagnosis":False}
