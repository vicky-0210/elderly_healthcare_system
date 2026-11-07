"""Stateful optical pulse and respiratory modulation views. Input is complete seconds."""
import numpy as np
from scipy.signal import butter,sosfilt,sosfilt_zi,lfilter

MORPH=16
MOD=["baseline_shift","log_pulse_range","interval_shift","resp_baseline","resp_amplitude","resp_interval",
     "resp_energy_baseline","resp_energy_amplitude","resp_energy_interval","pulse_rms","interval_s"]
QUALITY=["high_frequency_fraction","equal_sample_fraction","beat_staleness","rail_dwell_proxy"]
NAMES=[f"pulse_{i}" for i in range(MORPH)]+MOD+QUALITY


class IIR:
    def __init__(self,sos):
        self.sos=sos; self.zi=None

    def push(self,x):
        if self.zi is None: self.zi=sosfilt_zi(self.sos)*x[0]
        y,self.zi=sosfilt(self.sos,x,zi=self.zi)
        return y


class EWMA:
    def __init__(self,seconds):
        self.alpha=-np.expm1(-1/seconds); self.zi=None

    def push(self,x):
        if self.zi is None: self.zi=[(1-self.alpha)*x[0]]
        y,self.zi=lfilter([self.alpha],[1,self.alpha-1],x,zi=self.zi)
        return y


class OpticalViews:
    def __init__(self,fs=80,denoise=True):
        if fs not in (80,100): raise ValueError("Expected Mindray 80 Hz or MAX30102 100 Hz")
        self.fs=fs; self.denoise=denoise
        self.lp=IIR(butter(3,6,fs=fs,output="sos"))
        self.cardiac=IIR(butter(2,[.45,5.5],fs=fs,btype="bandpass",output="sos"))
        self.base=EWMA(45); self.amp=EWMA(45); self.rate=EWMA(45)
        self.resp=[IIR(butter(2,[.08,.38],fs=1,btype="bandpass",output="sos")) for _ in range(3)]
        self.energy=[EWMA(8) for _ in range(3)]
        self.last_value=0.; self.last_cross=None; self.last_good=None; self.period=.85; self.offset=0

    def periods(self,pulse):
        fs=self.fs
        flat=pulse.ravel(); crossings=np.flatnonzero((flat>0)&(np.r_[self.last_value,flat[:-1]]<=0))+self.offset
        self.last_value=flat[-1]
        ends=self.offset+fs*np.arange(1,len(pulse)+1)-1
        if len(crossings):
            pre=self.last_cross if self.last_cross is not None else crossings[0]
            ibi=np.diff(crossings,prepend=pre)/fs
            good=(ibi>=.30)&(ibi<=1.8)
            self.last_cross=crossings[-1]
        else: good=np.zeros(0,dtype=bool); ibi=np.empty(0)
        times=np.r_[self.last_good if self.last_good is not None else -(10**12),crossings[good]]
        vals=np.r_[self.period,ibi[good]]; loc=np.searchsorted(times,ends,side="right")-1
        age=np.minimum(4,(ends-times[loc])/fs)
        if good.any(): self.last_good=times[-1]; self.period=vals[-1]
        self.offset+=len(flat)
        return vals[loc],age

    def push(self,rows):
        a=np.asarray(rows,dtype=float)
        if a.ndim!=2 or a.shape[1]!=self.fs or not len(a) or not np.isfinite(a).all(): raise ValueError("Expected finite complete seconds")
        # Shrink finest Haar details only, preserving each two-sample mean exactly.
        even,odd=a[:,::2],a[:,1::2]
        mid=(even+odd)*.5; detail=(odd-even)*.5
        noise=np.median(np.abs(detail-np.median(detail,axis=1,keepdims=True)),axis=1,keepdims=True)
        detail=np.sign(detail)*np.maximum(np.abs(detail)-.5*noise,0)
        clean=np.stack((mid-detail,mid+detail),axis=-1).reshape(a.shape) if self.denoise else a
        smooth=self.lp.push(clean.ravel()).reshape(a.shape)
        pulse=self.cardiac.push(clean.ravel()).reshape(a.shape)
        dc=np.median(smooth,axis=1)
        lo,hi=np.percentile(smooth,[10,90],axis=1); amp=np.maximum(hi-lo,1e-5)
        scale=np.maximum(self.amp.push(amp),1e-5)
        baseline=(dc-self.base.push(dc))/scale
        amplitude=np.log(amp/scale)
        period,age=self.periods(pulse)
        interval=period-self.rate.push(period)
        bands=[f.push(v) for f,v in zip(self.resp,[baseline,amplitude,interval])]
        energies=[np.sqrt(np.maximum(f.push(v*v),0)) for f,v in zip(self.energy,bands)]
        # Sample the bandlimited waveform at 16 fixed positions per full second.
        at=(np.arange(MORPH)+.5)*self.fs/MORPH-.5
        left=at.astype(int); frac=at-left
        shape=(pulse[:,left]*(1-frac)+pulse[:,left+1]*frac)/scale[:,None]
        rms=np.sqrt(np.mean(pulse**2,axis=1))/scale
        residual=np.sqrt(np.mean((a-smooth)**2,axis=1))/scale
        equal=(np.diff(a,axis=1)==0).mean(1)
        dwell=np.maximum((a==a.min(1)[:,None]).mean(1),(a==a.max(1)[:,None]).mean(1))
        return np.column_stack([shape,baseline,amplitude,interval,*bands,*energies,rms,period,
            residual,equal,age,dwell]).astype("f")
