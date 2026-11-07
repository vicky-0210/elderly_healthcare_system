"""Stateful inference on successive calibrated acceleration (g) and gyroscope (deg/s) samples."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from wearable_common.inference import SignalPredictor
from wearable_imu.preprocessing.ipmv import IPMV,TOKEN_HZ
from .alerts import AlertPolicy


class IMUPredictor(SignalPredictor):
    def __init__(self,artifact_directory=None,trt_engine=None):
        if artifact_directory is not None: self.load(artifact_directory,trt_engine)

    def load(self,artifact_directory,trt_engine=None):
        root=Path(artifact_directory)
        self.meta=m=json.loads((root/"runtime.json").read_text())
        if m["format"]!=1: raise ValueError("Unsupported IMU export")
        self.fs=m["sample_rate_hz"]; self.width=m["width"]; self.hop=m["hop"]
        if m["engine"]=="numpy":
            if trt_engine: raise ValueError("The selected ensemble uses its native compact NumPy evaluator")
            from .portable import PortableClassifier
            self.infer=PortableClassifier(root/m["file"])
        else:
            self.center=np.asarray(m["center"],dtype="f"); self.scale=np.asarray(m["scale"],dtype="f")
            if trt_engine:
                from .tensorrt_runner import TensorRTWindow
                runner=TensorRTWindow(Path(trt_engine),len(m["channels"]),self.width)
                self.infer=lambda w:runner(np.arcsinh((w-self.center)/self.scale))
            else:
                import onnxruntime as ort
                opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
                self.session=ort.InferenceSession(str(root/m["file"]),sess_options=opts,providers=["CPUExecutionProvider"])
                self.infer=self.run_onnx
        self.reset()
        return self

    def run_onnx(self,windows):
        x=np.arcsinh((windows-self.center)/self.scale).astype("f")
        return np.array([self.session.run(None,{"window":b.T[None]})[0].item() for b in x])

    def reset(self):
        m=self.meta
        self.front=IPMV(**m["preprocessing"],waves=m["waves"],coefficients=m["wave_coefficients"])
        self.indices=[self.front.names.index(n) for n in m["channels"]]
        self.buffer=np.empty((0,len(self.indices)),dtype="f")
        self.origin=None; self.native=0; self.used=0
        self.alert=AlertPolicy(m["threshold"])

    def push(self,physical,first_time):
        x=np.asarray(physical)
        if x.ndim!=2 or x.shape[1:]!=(6,) or not np.isfinite(x).all() or not np.isfinite(first_time):
            self.reset(); raise ValueError("Invalid timestamps or six-axis measurement block")
        if not len(x): return []
        if self.origin is not None and abs(first_time-(self.origin+self.native/self.fs))>.51/self.fs:
            self.reset()
        if self.origin is None: self.origin=float(first_time)
        self.native+=len(x)
        new=self.front.push(x)[:,self.indices]
        self.buffer=np.concatenate((self.buffer,new))
        count=max(0,(len(self.buffer)-self.width)//self.hop+1)
        if not count: return []
        windows=np.stack([self.buffer[i*self.hop:i*self.hop+self.width] for i in range(count)])
        scores=self.infer(windows)
        result=[]
        for i,z in enumerate(scores):
            end=self.origin+(self.used+i*self.hop+self.width)/TOKEN_HZ
            a,b=self.meta["calibration"]
            result.append(dict(end_s=end,window_logit=float(z),fall_evidence=float(expit(a*z+b)),
                threshold_exceeded=bool(z>=self.meta["threshold"]),alert=self.alert.step(float(z),end)))
        self.buffer=self.buffer[count*self.hop:].copy(); self.used+=count*self.hop
        return result

    def predict(self,window):
        """Fresh recording in physical units. Use push for continuous acquisition."""
        if isinstance(window,dict):
            return self.predict_message(window)
        self.reset()
        rows=self.push(window,0.)
        if not rows: return dict(status="insufficient_signal",required_samples=self.fs*self.width//TOKEN_HZ)
        high=max(rows,key=lambda r:r["window_logit"])
        return dict(status="ok",trial_fall_probability=high["fall_evidence"],
            decision="FALL" if high["threshold_exceeded"] else "ADL",windows=len(rows))

    def predict_message(self,message):
        meta=getattr(self,"meta",None)
        result=dict(schema_version="1.0",signal="imu",stream_id=message.get("stream_id"),
            clock_domain=message.get("clock_domain"),model_version=meta["experiment"] if meta else None,
            preprocessing_version="sisfall-ipmv-v1",threshold_calibration_version="sisfall-oof-v1",quality_flags=[])
        if meta is None: return dict(result,status="model_unavailable")
        try:
            x=np.asarray(message["samples"],dtype=float); t=np.asarray(message["timestamps"],dtype=float)
            if (message.get("signal")!="imu" or message.get("channels")!=["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
                or message.get("units")!=meta["input_units"] or message.get("sample_rate_hz")!=self.fs):
                raise ValueError("incompatible_input_schema")
            if (t.ndim!=1 or x.shape!=(len(t),6) or len(t)<self.fs*self.width//TOKEN_HZ
                or not np.isfinite(t).all() or not np.isfinite(x).all()):
                raise ValueError("short_or_nonfinite")
            if np.any(np.abs(np.diff(t)-1/self.fs)>.51/self.fs): raise ValueError("noncontiguous_clock")
            if np.any(message.get("missing_mask",False)): raise ValueError("missing_samples")
            pred=self.predict(x)
            pos=pred["trial_fall_probability"]
            return dict(result,status="ok",window_start_s=float(t[0]),window_end_s=float(t[-1]+1/self.fs),
                scores=dict(ADL=1-pos,FALL=pos),decision=pred["decision"],score_semantics="recording maximum")
        except (KeyError,TypeError,ValueError) as exc:
            return dict(result,status="insufficient_signal",quality_flags=[str(exc)])
