"""Consume timestamped digital ADC blocks on stdin; write buffered research scores on stdout."""
import argparse
import json
import sys

import numpy as np
from wearable import WearableStudent, convert_counts


class BufferedADC:
    def __init__(self,student,fs,zero,mv_per_count,polarity,rail_min,rail_max):
        if fs not in (500,student.meta["hz"]) or rail_min>=rail_max:
            raise ValueError("Unsupported board configuration")
        convert_counts([zero],zero,mv_per_count,polarity)
        self.student,self.fs=student,fs
        self.calibration=zero,mv_per_count,polarity
        self.rails=rail_min,rail_max
        self.reset()

    def reset(self):
        self.times=np.empty(0);self.voltage=np.empty(0);self.last=None

    def push(self,counts,timestamps):
        t=np.asarray(timestamps,dtype=float); counts=np.asarray(counts,dtype=float)
        if not len(t) or t.ndim!=1 or counts.shape!=t.shape or not np.isfinite(t).all():
            self.reset();raise ValueError("Malformed ADC block")
        valid=np.isfinite(counts)&(counts>self.rails[0])&(counts<self.rails[1])
        broken=np.abs(np.diff(t,prepend=t[0]-(1/self.fs) if self.last is None else self.last)-1/self.fs)>.2/self.fs
        edges=np.unique(np.r_[0,np.flatnonzero(broken),np.flatnonzero(~valid),np.flatnonzero(~valid)+1,len(t)])
        output=[]
        for left,right in zip(edges[:-1],edges[1:]):
            if broken[left] or not valid[left]:
                self.reset();output.append(dict(status="insufficient_signal",reason="ADC rail, missing sample or clock gap",score=None))
            if not valid[left]:continue
            self.times=np.r_[self.times,t[left:right]]
            self.voltage=np.r_[self.voltage,convert_counts(counts[left:right],*self.calibration)]
            self.last=t[right-1]
            n=10*self.fs
            while len(self.times)>=n:
                row=self.student.predict(self.voltage[:n],self.times[:n],self.fs,lead="II",units="mV")
                row["window_start"]=self.times[0];row["window_end"]=self.times[n-1]+1/self.fs
                output.append(row)
                self.times=self.times[2*self.fs:];self.voltage=self.voltage[2*self.fs:]
        return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    for key in ("fs","polarity","rail-min","rail-max"):p.add_argument("--"+key,type=int,required=True)
    for key in ("zero","mv-per-count"):p.add_argument("--"+key,type=float,required=True)
    a=p.parse_args()
    stream=BufferedADC(WearableStudent(a.bundle),a.fs,a.zero,a.mv_per_count,a.polarity,a.rail_min,a.rail_max)
    for line in sys.stdin:
        try:
            packet=json.loads(line)
            for result in stream.push(packet["counts"],packet["timestamps"]):print(json.dumps(result),flush=True)
        except (TypeError,ValueError,KeyError) as err:
            stream.reset();print(json.dumps(dict(status="insufficient_signal",reason=str(err),score=None)),flush=True)
