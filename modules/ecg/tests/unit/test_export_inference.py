import numpy as np
import torch
from wearable_ecg.models.backbones import DiagonalSSM
from wearable_ecg.inference.export import FrozenSSM
from wearable_ecg.inference.predictor import ECGPredictor


def test_frozen_ssm_equivalence():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    block=DiagonalSSM(8,4).eval(); fixed=FrozenSSM(block,length=300).eval()
    x=torch.randn(2,8,300)
    with torch.no_grad(): torch.testing.assert_close(block(x),fixed(x),rtol=1e-4,atol=1e-5)


def test_fixed_quality_pool_matches_training_adaptive_pool():
    x=torch.rand(4,1,3600)
    torch.testing.assert_close(torch.nn.functional.adaptive_avg_pool1d(x,300),
                               torch.nn.functional.avg_pool1d(x,12,stride=12),rtol=0,atol=0)


def test_rejects_uncalibrated_and_gapped_input():
    p=ECGPredictor(); p.loaded=True
    bad=p.predict({"samples":np.ones(3600),"sampling_rate_hz":360,"lead":"II","units":"ADC counts"})
    assert bad["score"] is None
    time=np.arange(3600)/360; time[1500:]+=.1
    bad=p.predict({"samples":np.ones(3600),"sampling_rate_hz":360,"lead":"II","units":"mV","timestamps":time})
    assert bad["status"]=="insufficient_signal"
