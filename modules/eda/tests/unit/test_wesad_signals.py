"""Regression tests for causal replay, independent modalities and calibration."""
import unittest
import numpy as np

from wearable_eda.preprocessing.features import window_features
from wearable_eda.preprocessing.wrist import transform_recording
from wearable_eda.datasets.wesad import _eligible


def sample_streams(n=1000):
    t=np.arange(n)/4
    rng=np.random.default_rng(810)
    return dict(eda=2+.08*np.sin(t/6)+.01*rng.normal(size=n),
                acc=rng.normal(size=(8*n,3))+[0.,0.,64.],
                temp=32+.6*np.sin(t/120),
                bvp=np.sin(2*np.pi*1.1*np.arange(16*n)/64))


def prefix(streams, a, b):
    return {k:v[int(a*m):int(b*m)] for k,v in streams.items()
            for m in [{"eda":1,"acc":8,"temp":1,"bvp":16}[k]]}


class WristSignals(unittest.TestCase):
    def test_complete_bin_prefix_invariance(self):
        raw=sample_streams()
        for method in ("mcsaed","sparse"):
            for mode in ("zero","rest"):
                kwargs=dict(modalities=tuple(raw),decomposition=method,calibration=mode,rest=(0,480))
                full,names=transform_recording(raw,**kwargs)
                part,ns=transform_recording(prefix(raw,0,793),**kwargs)
                self.assertEqual(ns,names)
                np.testing.assert_allclose(part[-240:],full[793-240:793],atol=1e-6)
                f,fn=window_features(part[-240:].T,ns)
                expected,en=window_features(full[793-240:793].T,names)
                np.testing.assert_allclose(f,expected,atol=1e-6)
                self.assertEqual(fn,en)

    def test_eda_only_independent_of_other_streams(self):
        raw=sample_streams()
        x,names=transform_recording(raw)
        raw["acc"][:]=np.nan
        raw["bvp"][:]=9000
        raw["temp"][:]=-100
        result,_=transform_recording(raw)
        np.testing.assert_equal(x,result)
        self.assertTrue((x[:,names.index("relative_tonic")]==0).all())
        with self.assertRaises(ValueError):
            transform_recording(raw,quality="motion")

    def test_rest_statistics_use_only_designated_capture(self):
        raw=sample_streams()
        interval=(100,580)
        x,names=transform_recording(raw,calibration="rest",rest=interval)
        rest,_=transform_recording(prefix(raw,*interval))
        for key,floor in (("tonic",.05),("phasic",.02)):
            k=names.index(key)
            rv=rest[rest[:,names.index("quality")]>.4,k]
            m=np.median(rv)
            expected=(x[700,k]-m)/max(floor,1.4826*np.median(abs(rv-m)))
            self.assertAlmostEqual(expected,x[700,names.index("relative_"+key)],places=4)
        with self.assertRaises(ValueError):
            transform_recording(raw,calibration="rest")

    def test_missing_segments_excluded_and_shapes_checked(self):
        raw=sample_streams()
        raw["eda"][520:545]=np.nan
        missing,longs=_eligible(raw["eda"])
        self.assertEqual(missing[-1],25)
        self.assertEqual(longs[-1],25)
        x,names=transform_recording(raw)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[520:545,names.index("missing")]==1).all())
        fresh,_=transform_recording(prefix(raw,545,1000))
        np.testing.assert_allclose(x[545:],fresh,atol=1e-6)
        with self.assertRaises(ValueError):
            transform_recording(dict(raw,acc=raw["acc"][:-1]),modalities=("eda","acc"))

    def test_motion_alone_does_not_mark_scr_as_corruption(self):
        raw=sample_streams()
        raw["eda"][:]=2+0.02*np.sin(np.arange(len(raw["eda"]))/5)
        raw["acc"]*=50
        x,n=transform_recording(raw,modalities=("eda","acc"))
        self.assertGreater(np.median(x[:,n.index("motion")]),.9)
        self.assertGreater(np.mean(x[:,n.index("quality")]),.99)

if __name__=="__main__":
    unittest.main()

