import sys
from pathlib import Path
import numpy as np
import pytest

repo=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(repo/"modules/imu/src"),str(repo/"acquisition/src")]
from wearable_imu.datasets.sisfall import read_counts,COUNTS_TO_PHYSICAL
from wearable_imu.datasets.views import window_index
from wearable_imu.preprocessing.ipmv import IPMV,view_names,wave_coeffs,CausalDWT
from wearable_imu.inference.alerts import AlertPolicy
from wearable_acquisition.drivers.mpu9250.calibration import fit_calibration,apply
from wearable_acquisition.drivers.mpu9250.device import MPU9250


def test_release_format_and_units():
    row=read_counts(b"256,0,0,16384,0,0,712,888,999;\r\n\t")
    np.testing.assert_allclose(row[0,:6]*COUNTS_TO_PHYSICAL,[1,0,0,1000,0,0])
    with pytest.raises(ValueError): read_counts(b"1,2,3,4,5,6;")
    with pytest.raises(ValueError): read_counts(b"0,0,0,0,0,0,0,0,0;1")


@pytest.mark.parametrize("fs",[100,200])
def test_prefix_and_arbitrary_chunks(fs):
    x=np.random.default_rng(4).normal(size=(1361,6))
    base=IPMV(fs=fs).push(x)
    stream=IPMV(fs=fs); fragmented=np.concatenate([stream.push(b) for b in np.array_split(x,[1,9,12,48,67,303,988])])
    np.testing.assert_allclose(base,fragmented,atol=3e-5,rtol=1e-5)
    bad=x.copy(); bad[870:]+=99
    np.testing.assert_allclose(base[:870//(fs//20)],IPMV(fs=fs).push(bad)[:870//(fs//20)])


def test_rotation_and_impact_bypass():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(7); x=rng.normal(size=(630,6)); x[221,:3]=[13.,-3.,2.]
    front=IPMV(); original=front.push(x); selected=view_names("ipmv",front.names); ix=[front.names.index(n) for n in selected]
    y=(x.reshape(-1,2,3)@Rotation.random(random_state=rng).as_matrix()).reshape(-1,6)
    np.testing.assert_allclose(IPMV().push(y)[:,ix],original[:,ix],rtol=2e-5,atol=3e-5)
    assert original[22,front.names.index("a_peak")]==pytest.approx(np.linalg.norm(x[221,:3]))


def test_wavelet_coefficients_match_reference():
    from scipy.signal import lfilter,lfilter_zi
    x=np.random.default_rng(6).normal(size=(160,2))
    lo,hi=wave_coeffs("db2"); bands=CausalDWT((lo,hi)).apply(x)
    positions=np.arange(len(x)); level=x
    for i in range(3):
        filt=lambda b:lfilter(b,[1.],level,axis=0,zi=lfilter_zi(b,[1.])[:,None]*level[0])[0][1::2]
        detail=filt(hi); level=filt(lo); positions=positions[1::2]
        np.testing.assert_allclose(bands[i][positions],detail,atol=1e-10)
    np.testing.assert_allclose(bands[-1][positions],level,atol=1e-10)


def test_window_boundaries_and_bags():
    length=np.array([200,460]); offset=np.array([0,200]); start,g,bags,w=window_index(length,offset,2,.75)
    assert np.all(start>=offset[g]); assert np.all(start+w<=offset[g]+length[g])
    assert np.all(g[bags]==np.arange(2)[:,None])
    assert bags.shape==(2,8)


def test_alert_excursion_refractory_and_rearm():
    policy=AlertPolicy(.5)
    assert [i for i,x in enumerate([1,1,0,0,1,1,1,0,0,0,1]) if policy.step(x,i)]==[0,5,10]


def test_calibration():
    poses=np.repeat(np.eye(3),2,axis=0)*np.tile([1,-1],3)[:,None]
    noisy=np.tile((poses*[1.05,.97,1.01]+[.02,.03,-.01])[:,None,:],(1,140,1))
    rest=np.tile([1.3,-2.,.1],(140,1)); cal=fit_calibration(noisy,rest)
    test=apply(np.column_stack((noisy[:,0],rest[:6])),cal)
    np.testing.assert_allclose(test[:,:3],poses,atol=1e-9); np.testing.assert_allclose(test[:,3:],0,atol=1e-9)
    with pytest.raises(ValueError): fit_calibration(noisy[[0,0,0,0,0,0]],rest)


class FakeBus:
    def __init__(self):
        self.reg={0x75:0x71}; self.raw=np.array([2048,0,0,164,0,0]*3,dtype=">i2").tobytes(); self.offset=0
    def write_byte_data(self,a,r,v): self.reg[r]=v
    def read_byte_data(self,a,r): return self.reg.get(r,0)
    def read_i2c_block_data(self,a,r,n):
        if r==0x72: return (len(self.raw)-self.offset).to_bytes(2,"big")
        assert r==0x74; b=self.raw[self.offset:self.offset+n]; self.offset+=n; return b


def test_fifo_conversion_timing_and_overflow():
    bus=FakeBus(); dev=MPU9250(bus,clock=lambda:10.)
    t,x,status=dev.read()
    assert status=="ok" and t==pytest.approx(9.99)
    np.testing.assert_allclose(x,[[1,0,0,10,0,0]]*3)
    assert dev.read()[2]=="waiting"
    bus.reg[0x3a]=0x10; assert dev.read()[2]=="fifo_overflow"
    assert dev.origin is None
