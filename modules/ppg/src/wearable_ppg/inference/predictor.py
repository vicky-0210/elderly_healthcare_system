"""Single-optical-channel streaming replay. Research annotations never enter this path."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax
from wearable_common.inference import SignalPredictor
from wearable_ppg.preprocessing.optical import NAMES,OpticalViews


class PPGPredictor(SignalPredictor):
    def __init__(self,directory=None,fs=80,polarity=1):
        self.fs=fs
        if polarity not in (-1,1): raise ValueError("Polarity must be +1 or -1")
        self.polarity=polarity
        if directory is not None: self.load(directory)

    def load(self,artifact_directory):
        import onnxruntime as ort
        path=Path(artifact_directory)
        self.meta=json.loads((path/"runtime.json").read_text())
        if self.meta["features"] != NAMES or self.meta["hop"] != 5 or self.meta["context"] < 5:
            raise ValueError("Incompatible optical features or timing in model bundle")
        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path/"model.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.center=np.array(self.meta["center"],dtype="f"); self.scale=np.array(self.meta["scale"],dtype="f")
        self.reset(); return self

    def reset(self):
        self.front=OpticalViews(self.fs)
        self.history=[]; self.next_time=None; self.samples=np.empty(0); self.start=None; self.counter=0; self.above=0

    def push(self,values,first_time):
        data=np.asarray(values,dtype=float)
        if data.ndim!=1 or not np.isfinite(first_time): self.reset(); raise ValueError("Expected one optical stream and a finite timestamp")
        if self.next_time is not None and abs(first_time-self.next_time)>.51/self.fs: self.reset()
        if self.start is None: self.start=float(first_time)
        self.next_time=first_time+len(data)/self.fs
        packet=np.concatenate((self.samples,data)); n=len(packet)//self.fs
        self.samples=packet[n*self.fs:].copy(); output=[]
        for i in range(n):
            x=packet[i*self.fs:(i+1)*self.fs]
            begin=self.start; self.start+=1
            rails=self.fs==100 and np.mean((x==0)|(x==262143))>=.10
            if not np.isfinite(x).all() or np.ptp(x)==0 or rails:
                self.front=OpticalViews(self.fs); self.history=[]; self.counter=0; self.above=0
                output.append(dict(status="insufficient_signal",timestamp=begin)); continue
            f=self.front.push((self.polarity*x)[None])[0]
            window=self.meta["context"]
            self.history=(self.history+[f])[-window:]; self.counter+=1
            warmup=self.meta.get("warmup",window)
            if self.counter<warmup or (self.counter-warmup)%5: continue
            inp=np.arcsinh((np.asarray(self.history)-self.center)/self.scale)
            binary=self.meta.get("task")=="binary_event"
            if binary:
                # Match standardized zero padding and the explicit history mask used in training.
                inp=np.column_stack((inp,np.ones(len(inp),dtype="f")))
                inp=np.pad(inp,((window-len(inp),0),(0,0)))
            raw=self.session.run(None,{"views":inp[None]})[0][0]
            a,b=self.meta["calibration"]; proba=expit(a*raw[:,0]+b)
            subs=None if binary else softmax(raw[:,1:],axis=1)
            for j,p in enumerate(proba):
                active=bool(p>=self.meta["threshold"])
                self.above=self.above+1 if active else 0
                row=dict(status="ok",timestamp=begin-4+j,available_at=begin+1,p_event=float(p),
                         label="ANY_EVENT" if active else "NONE",above_threshold=active,new_alert=self.above==5)
                if subs is not None:
                    row["subtype_given_event"]=subs[j].tolist()
                    if active: row["label"]=self.meta["labels"][1+int(np.argmax(subs[j]))]
                output.append(row)
        return output

    def predict(self,window):
        """Predict one fresh recording expressed in optical samples at self.fs."""
        if isinstance(window,dict):
            return self.predict_message(window)
        self.reset()
        return self.push(window,0.)

    def predict_message(self,message):
        meta=getattr(self,"meta",None)
        result=dict(schema_version="1.0",signal="ppg",stream_id=message.get("stream_id"),
                    clock_domain=message.get("clock_domain"),model_version=meta["model"] if meta else None,
                    preprocessing_version="osasud_v1",threshold_calibration_version="osasud_v1_oof",quality_flags=[])
        if meta is None:
            return dict(result,status="model_unavailable")
        try:
            channels=message["channels"]
            allowed=(["pleth"],) if self.fs==80 else (["ir"],["red"])
            units=["monitor_units"] if self.fs==80 else ["adc_counts"]
            if (message["signal"]!="ppg" or channels not in allowed or message["units"]!=units
                    or message["sample_rate_hz"]!=self.fs):
                raise ValueError("incompatible_optical_stream")
            t=np.asarray(message["timestamps"],dtype=float)
            x=np.asarray(message["samples"],dtype=float)
            if t.ndim!=1 or x.shape!=(len(t),1) or len(t)<self.fs*meta.get("warmup",meta["context"]):
                raise ValueError("incomplete_window")
            if not np.isfinite(t).all() or np.any(np.abs(np.diff(t)-1/self.fs)>.51/self.fs):
                raise ValueError("discontinuous_clock")
            mask=np.asarray(message.get("missing_mask",False),dtype=bool)
            if mask.any() or not np.isfinite(x).all():
                raise ValueError("missing_optical_samples")
            self.reset()
            rows=self.push(x[:,0],t[0])
            if not rows or any(row["status"]!="ok" for row in rows):
                raise ValueError("unusable_signal")
            semantics=("one binary event probability per scored second" if meta.get("task")=="binary_event"
                       else "one event probability and four conditional subtype scores per scored second")
            result["threshold_calibration_version"]=meta.get("calibration_version","osasud_v1_oof")
            return dict(result,status="ok",window_start_s=float(t[0]),window_end_s=float(t[-1]+1/self.fs),
                        scores=rows,score_semantics=semantics)
        except (KeyError,TypeError,ValueError) as exc:
            return dict(result,status="insufficient_signal",quality_flags=[str(exc)])
