"""Inference on a buffered, calibrated ten-second Lead-II-equivalent recording."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.special import expit
from wearable_ecg.preprocessing.pipeline import resample_mv, process, landmarks


def convert_counts(counts, zero_count, mv_per_count, polarity):
    """Use coefficients measured at the electrode input through the complete analog chain."""
    if not np.isfinite([zero_count,mv_per_count]).all() or mv_per_count<=0 or polarity not in (-1,1):
        raise ValueError("Invalid bench calibration")
    return polarity*(np.asarray(counts,dtype=np.float64)-zero_count)*mv_per_count


class WearableStudent:
    def __init__(self,directory):
        import joblib
        import onnxruntime as ort
        root=Path(directory)
        self.meta=json.loads((root/"contract.json").read_text())
        self.net=ort.InferenceSession(str(root/"model.onnx"),providers=["CPUExecutionProvider"])
        self.controller=joblib.load(root/"controller.joblib") if self.meta["pp"]=="constrained" else None

    def predict(self,samples,timestamps,fs,lead="II",units="mV",clipped=False):
        x=np.asarray(samples,dtype=np.float64); t=np.asarray(timestamps,dtype=float)
        def invalid(reason): return dict(status="insufficient_signal",reason=reason,score=None)
        rate=self.meta["hz"]
        if lead!="II" or units!="mV" or fs not in (500,rate):return invalid("incompatible calibration, placement or sample rate")
        if x.shape!=(10*fs,) or t.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(t).all():return invalid("incomplete or nonfinite input")
        if np.any(np.abs(np.diff(t)-1/fs)>.2/fs) or clipped or np.std(x)<1e-5:return invalid("clock gap, clipping or flat input")
        x=resample_mv(x,fs,rate) if fs!=rate else x.astype("f")
        pp=self.meta["pp"];f=np.zeros(16,"f"); beat=np.zeros(144,"f")
        if pp in ("constrained","conventional_fixed"):
            x,f,q,_=process(x,self.controller,pp)
            if np.mean(q)<.5:return invalid("low signal quality")
        elif pp=="drift":x=x-ndimage.uniform_filter1d(x,217,mode="reflect")
        elif pp=="mains":
            z=np.fft.rfft(x);freq=np.fft.rfftfreq(len(x),1/rate)
            for hz in (50,60):
                band=abs(freq-hz)<.15
                if np.mean(abs(z[band])**2)>8*np.mean(abs(z[(abs(freq-hz)>1)&(abs(freq-hz)<4)])**2):z[band]=0
            x=np.fft.irfft(z).astype("f")
        if self.meta.get("fusion"):
            p=landmarks(x);p=p[(p>=54)&(p+90<=len(x))]
            if len(p):beat=np.median(x[p[:,None]+np.arange(-54,90)],axis=0)
        inputs=dict(wave=x.reshape(1,1,-1),features=f[None],beat=beat.reshape(1,1,-1))
        z=self.net.run(None,{v.name:inputs[v.name] for v in self.net.get_inputs()})[0].item()
        p=float(expit(self.meta["slope"]*z+self.meta["intercept"]))
        return dict(status="research_score",score=p,flag=p>=self.meta["threshold"],logit=z)
