"""Causal, chunk-invariant inertial processing in g and degrees/s."""
from functools import lru_cache
import numpy as np
from scipy import signal

TOKEN_HZ=20
AXES=["a_x","a_y","a_z","g_x","g_y","g_z"]
IMPACT=["a_mean","a_peak","a_min","a_jerk","a_filt_rms","g_mean","g_peak","g_change","g_filt_rms"]
BODY=["a_gravity","a_dynamic","a_vertical","a_horizontal","a_tilt","g_vertical","g_horizontal"]
DEFAULT=dict(fs=200,acc_hz=30,gyro_hz=30,order=2,adaptive=True,gravity="butter")


@lru_cache(None)
def butter(fs,cutoff,order):
    sos=signal.butter(order,cutoff,fs=fs,output="sos")
    return sos,signal.sosfilt_zi(sos)


class LowPass:
    def __init__(self,fs,cutoff,order):
        self.sos,self.init=butter(fs,cutoff,order)
        self.state=None

    def apply(self,x):
        if self.state is None:
            self.state=self.init[:,:,None]*x[0]
        y,self.state=signal.sosfilt(self.sos,x,axis=0,zi=self.state)
        return y


class FIR:
    def __init__(self,b):
        self.b=np.asarray(b,dtype=float)
        self.state=None

    def apply(self,x):
        if self.state is None:
            self.state=np.cumsum(self.b[:0:-1])[::-1,None]*x[0]
        y,self.state=signal.lfilter(self.b,[1.],x,axis=0,zi=self.state)
        return y


class CausalDWT:
    """Critically sampled FIR analysis; emit held coefficients only after arrival."""
    def __init__(self,coefficients,levels=3):
        lo,hi=coefficients
        self.banks=[(FIR(lo),FIR(hi)) for _ in range(levels)]
        self.received=[0]*levels
        self.last=[None]*(levels+1)

    def align(self,coeff,pos,n,index,initial):
        previous=self.last[index] if self.last[index] is not None else initial
        entries=np.vstack((previous,coeff))
        aligned=entries[np.searchsorted(pos,np.arange(n),side="right")]
        self.last[index]=entries[-1]
        return aligned

    def apply(self,values):
        n=len(values)
        pos=np.arange(n)
        x=values
        result=[]
        for i,(lo,hi) in enumerate(self.banks):
            if len(x):
                start=1-self.received[i]%2
                self.received[i]+=len(x)
                detail=hi.apply(x)[start::2]
                x=lo.apply(x)[start::2]
                pos=pos[start::2]
            else:
                detail=x
            result.append(self.align(detail,pos,n,i,np.zeros(2)))
        result.append(self.align(x,pos,n,len(self.banks),values[0]))
        return result


@lru_cache(None)
def wave_coeffs(name):
    import pywt
    wave=pywt.Wavelet(name)
    return np.array(wave.dec_lo)/np.sqrt(2),np.array(wave.dec_hi)/np.sqrt(2)


def channel_names(waves):
    result=AXES+["a_sma"]+IMPACT+BODY
    for wave in waves:
        for band in ("d1","d2","d3","l3"):
            result += [f"a_{wave}_{band}",f"g_{wave}_{band}"]
    return result


def view_names(view,names):
    if view=="axes":
        return AXES
    if view=="axes_ipmv":
        return AXES+["a_sma"]+view_names("ipmv",names)
    if view=="impact": return IMPACT
    if view=="body": return BODY
    if view=="no_peaks":
        return [n for n in view_names("ipmv",names) if n not in ("a_peak","a_min","a_jerk","g_peak","g_change")]
    if view in ("acc","gyro"):
        return [n for n in view_names("ipmv",names) if n.startswith("a_" if view=="acc" else "g_")]
    if view=="no_wave": return IMPACT+BODY
    if view=="no_body": return IMPACT+[n for n in names if "_db2_" in n]
    if view in ("ipmv","sym4","wave2"):
        wave="sym4" if view=="sym4" else "db2"
        return IMPACT+BODY+[n for n in names if f"_{wave}_" in n and (view!="wave2" or n.endswith(("d1","d2")))]
    raise ValueError(f"Unknown inertial view {view}")


class IPMV:
    def __init__(self, *, fs=200,acc_hz=30,gyro_hz=30,order=2,adaptive=True,gravity="butter",waves=("db2","sym4"),coefficients=None):
        if fs not in (100,200) or order not in (2,4) or not all(0<=hz<fs/2 for hz in (acc_hz,gyro_hz)):
            raise ValueError("Invalid sample rate or filter design")
        if gravity not in ("butter","mean"):
            raise ValueError("Unknown gravity estimator")
        self.fs,self.width,self.adaptive=fs,fs//TOKEN_HZ,adaptive
        self.filters=[LowPass(fs,hz,order) if hz else None for hz in (acc_hz,gyro_hz)]
        self.gravity=LowPass(fs,.75,2) if gravity=="butter" else FIR(np.ones(int(fs*.75))/(fs*.75))
        self.names=channel_names(waves)
        self.wavelets=[CausalDWT(coefficients[w] if coefficients else wave_coeffs(w)) for w in waves]
        self.pending=np.empty((0,6))
        self.last_magnitude=None

    def push(self,physical):
        a=np.asarray(physical,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():
            raise ValueError("Expected finite six-axis samples")
        x=np.concatenate((self.pending,a))
        whole=len(x)//self.width*self.width
        self.pending=x[whole:].copy()
        if not whole: return np.empty((0,len(self.names)),dtype="f")
        x=x[:whole]
        acc,gyro=x[:,:3],x[:,3:]
        mags=np.stack((np.linalg.norm(acc,axis=1),np.linalg.norm(gyro,axis=1)),axis=1)
        prev=self.last_magnitude if self.last_magnitude is not None else mags[0]
        delta=np.abs(np.diff(mags,axis=0,prepend=prev[None]))*self.fs
        self.last_magnitude=mags[-1].copy()
        smooth=[]
        for part,smoother,floor in zip((acc,gyro),self.filters,(.35,50.)):
            y=smoother.apply(part) if smoother else part
            if self.adaptive and smoother:
                innovation=part-y
                size=np.linalg.norm(innovation,axis=1,keepdims=True)
                y=y+innovation*size/(size+floor)
            smooth.append(y)
        h=self.gravity.apply(smooth[0])
        hnorm=np.linalg.norm(h,axis=1)
        axis=h/np.maximum(hnorm[:,None],1e-8)
        dynamic=smooth[0]-h
        vertical=np.einsum("ij,ij->i",dynamic,axis)
        horiz=np.linalg.norm(dynamic-vertical[:,None]*axis,axis=1)
        gvert=np.einsum("ij,ij->i",smooth[1],axis)
        ghoriz=np.linalg.norm(smooth[1]-gvert[:,None]*axis,axis=1)
        def bins(v): return v.reshape(-1,self.width)
        def rms(v): return np.sqrt(np.mean(bins(v)**2,axis=1))
        head=x.reshape(-1,self.width,6).mean(1)
        a,g=bins(mags[:,0]),bins(mags[:,1])
        tilt=1-np.clip(np.einsum("ij,ij->i",axis[::self.width],axis[self.width-1::self.width]),-1,1)
        cols=[head,bins(np.abs(acc).sum(1)).mean(1),a.mean(1),a.max(1),a.min(1),bins(delta[:,0]).max(1),rms(np.linalg.norm(smooth[0],axis=1)),
              g.mean(1),g.max(1),bins(delta[:,1]).max(1),rms(np.linalg.norm(smooth[1],axis=1)),
              bins(hnorm).mean(1),rms(np.linalg.norm(dynamic,axis=1)),bins(vertical).mean(1),bins(horiz).max(1),
              tilt,rms(gvert),rms(ghoriz)]
        for wave in self.wavelets:
            for band in wave.apply(mags):
                cols.extend((rms(band[:,0]),rms(band[:,1])))
        return np.column_stack(cols).astype("f")
