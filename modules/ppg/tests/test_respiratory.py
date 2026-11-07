from pathlib import Path
import sys
import numpy as np
import pytest
repo=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(repo/"modules/ppg/src"),str(repo/"acquisition/src"),str(repo/"packages/common/src")]
from wearable_ppg.preprocessing.optical import OpticalViews,NAMES
from wearable_ppg.datasets.windows import spans
from wearable_ppg.evaluation.metrics import match_events
from wearable_ppg.inference.predictor import PPGPredictor
from wearable_acquisition.drivers.max30102.device import MAX30102


@pytest.mark.parametrize("fs",[80,100])
def test_optical_chunk_equivalence(fs):
    t=np.arange(fs*63)/fs
    x=(100+(20+5*np.sin(t))*np.sin(7*t)+3*np.sin(1.2*t)).reshape(-1,fs)
    full=OpticalViews(fs).push(x); front=OpticalViews(fs)
    fragments=np.concatenate([front.push(b) for b in np.split(x,[1,5,21])])
    assert full.shape==(63,len(NAMES))
    np.testing.assert_allclose(full,fragments,rtol=5e-5,atol=5e-5)
    x[34:]+=50
    np.testing.assert_allclose(full[:34],OpticalViews(fs).push(x)[:34])


def test_boundaries_and_matching():
    t=np.r_[np.arange(8),np.arange(10,15)]
    valid=np.ones(len(t),bool); valid[3]=False
    assert [len(s) for s in spans(t,valid)]==[3,4,5]
    t=np.arange(30); truth=(t>=3)&(t<20); pred=((t>=2)&(t<9))|((t>=11)&(t<19))|(t>=28)
    tp,fp,fn,delay=match_events(truth,pred,t,t+1)
    assert (tp,fp,fn)==(1,1,0) and delay==[13]


class MockSession:
    def run(self,_,inputs): return [np.zeros((1,5,5),dtype="f")]


def test_alarm_timestamp_and_gap():
    p=PPGPredictor(); p.meta=dict(context=30,threshold=.4,calibration=[1.,0.],labels=["NONE","H","C","O","M"])
    p.session=MockSession(); p.center=np.zeros(len(NAMES)); p.scale=np.ones(len(NAMES)); p.reset()
    x=np.tile(np.sin(np.arange(80)/12),32)
    rows=p.push(x,20.)
    assert [r["timestamp"] for r in rows]==list(range(45,50))
    assert [r["new_alert"] for r in rows]==[False]*4+[True]
    assert rows[0]["available_at"]==50
    assert p.push(x[:80],88.)==[]
    assert p.push(np.zeros(80),89.)[0]["status"]=="insufficient_signal"


class Bus:
    def __init__(self): self.reg={255:0x15}; self.over=0
    def write_byte_data(self,a,r,v): self.reg[r]=0 if r==9 and v==64 else v
    def read_byte_data(self,a,r): return self.reg.get(r,0)
    def read_i2c_block_data(self,a,r,n):
        return [2,self.over,0] if r==4 else [0,1,2,3,255,254]*2


def test_fifo_red_ir_and_overflow():
    bus=Bus(); dev=MAX30102(bus,clock=lambda:15.)
    stamp,rows,status=dev.poll()
    assert stamp==14.99 and status=="ok"
    np.testing.assert_equal(rows,[[258,262142]]*2)
    bus.over=1; assert dev.poll()[2]=="fifo_gap"


def test_contract_missing_model_gap_and_scores():
    p=PPGPredictor()
    msg=dict(signal="ppg",channels=["pleth"],units=["monitor_units"],sample_rate_hz=80,
             stream_id="replay",clock_domain="relative",timestamps=np.arange(2400)/80,
             samples=np.sin(np.arange(2400)/10)[:,None])
    assert p.predict(msg)["status"]=="model_unavailable"
    p.meta=dict(model="fixture",context=30,threshold=.4,calibration=[1.,0.],labels=["NONE","H","C","O","M"])
    p.session=MockSession(); p.center=np.zeros(len(NAMES)); p.scale=np.ones(len(NAMES))
    assert len(p.predict(msg)["scores"])==5
    msg["timestamps"][10:]+=.1
    assert p.predict(msg)["quality_flags"]==["discontinuous_clock"]
