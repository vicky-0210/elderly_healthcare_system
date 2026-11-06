import numpy as np
import pandas as pd
import pytest
import torch
from wearable_ecg.labels.ptbxl import annotate,cohort,assert_patient_folds
from wearable_ecg.preprocessing.pipeline import process, resample_mv, correct
from wearable_ecg.models.backbones import ECGClassifier,DiagonalSSM
from wearable_ecg.evaluation.metrics import choose_threshold,metrics,calibrated,fit_calibration
from wearable_ecg.runtime import separate_output


def test_exact_statement_and_contradiction():
    f=pd.DataFrame({"ecg_id":[1,2,3,4],"patient_id":[1,2,3,4],"age":[60,80,300,50],"strat_fold":[1,7,9,10],
                    "scp_codes":["{'LVH': 0}","{'NORM':100,'SR':0}","{'NORM':100,'LVH':100}","{'RVH':100}"]})
    a=annotate(f)
    assert cohort(a,"lvh_vs_normal").ecg_id.tolist()==[1,2]
    assert cohort(a,"lvh_vs_rest").ecg_id.tolist()==[1,2,4]
    assert a.age_for_control.iloc[2]==90
    assert_patient_folds(a)
    a.loc[1,"patient_id"]=1
    with pytest.raises(ValueError): assert_patient_folds(a)


def test_source_guard(tmp_path):
    with pytest.raises(ValueError): separate_output(tmp_path,tmp_path/"derived")


def test_resampling_retains_physical_voltage():
    t=np.arange(5000)/500
    x=(1.5*np.sin(2*np.pi*10*t)).astype(np.float32)
    y=resample_mv(x)
    assert len(y)==3600
    assert abs(np.std(y[100:-100])-np.std(x[140:-140]))<.015
    a,features,q,c=process(y,variant="minimal")
    np.testing.assert_array_equal(a,y)
    assert features.shape==(16,) and q.shape==y.shape
    with pytest.raises(ValueError): process(np.full(3600,np.nan))


@pytest.mark.parametrize("name",["cnn","cnn_transformer","cnn_ssm","cnn_bigru"])
def test_model_gradient_and_serialization(name):
    torch.set_num_threads(2)
    m=ECGClassifier(name)
    x=torch.randn(2,3600); f=torch.randn(2,16)
    out=m(x,f); assert out.shape==(2,)
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert sum(p.numel() for p in m.parameters())<1_000_000


def test_ssm_fft_matches_direct_convolution():
    m=DiagonalSSM(2,4)
    x=torch.randn(1,2,16)
    k=m.kernel(16)
    fft=torch.fft.irfft(torch.fft.rfft(x,n=32)*torch.fft.rfft(k,n=32)[None],n=32)[...,:16]
    direct=torch.nn.functional.conv1d(torch.nn.functional.pad(x,(15,0)),k.flip(-1)[:,None],groups=2)
    torch.testing.assert_close(fft,direct,rtol=1e-4,atol=1e-5)


def test_tied_scores_and_calibration():
    y=np.array([0]*20+[1]*10); p=np.array([.2]*18+[.6]*2+[.6]*10)
    t=choose_threshold(y,p,.90)
    assert metrics(y,p,t)["specificity"]>=.90
    logits=np.linspace(-3,3,len(y)); c=fit_calibration(logits,y)
    assert c["slope"]>0
    assert np.all(np.diff(calibrated(logits,c))>=0)
