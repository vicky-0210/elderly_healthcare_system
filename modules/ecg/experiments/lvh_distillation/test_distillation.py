import numpy as np
import pytest
import torch
from network import MorphNet
from wearable import convert_counts


def test_masked_teacher_leads_cannot_change_prediction():
    torch.manual_seed(42)
    net=MorphNet({"teacher":"lead_attention"}).eval()
    x=torch.randn(2,12,3600); f=torch.zeros(2,16); beats=torch.zeros(2,1,144)
    mask=torch.ones(2,12,dtype=torch.bool);mask[:,3]=False
    with torch.inference_mode():
        before=net(x,f,beats,mask)[0]
        x[:,3]=x[:,3]*77+100
        torch.testing.assert_close(before,net(x,f,beats,mask)[0])


@pytest.mark.parametrize("hz",[100,360])
def test_raw_student_does_not_require_teacher_targets(hz):
    net=MorphNet({"hz":hz}).eval()
    with torch.inference_mode():
        x=torch.rand(2,1,10*hz); f=torch.zeros(2,16); b=torch.zeros(2,1,144)
        a=net(x,f,b)[0]
        torch.testing.assert_close(a,net(x,f+100,b+100)[0])
        assert a.shape==(2,)


def test_measured_adc_gain_and_polarity():
    np.testing.assert_allclose(convert_counts([2048,2148],2048,.002,-1),[0,-.2])
    with pytest.raises(ValueError):convert_counts([400],0,0,1)


def test_rolling_adc_hop_and_restart():
    from bioamp_stream import BufferedADC
    class Stub:
        meta={"hz":100}
        def predict(self,x,t,fs,lead,units):
            assert np.allclose(x,1.) and len(x)==1000 and lead=="II" and units=="mV"
            return dict(status="research_score")
    stream=BufferedADC(Stub(),100,1000,.01,1,0,4095)
    t=np.arange(1500)/100
    rows=[]
    for chunk in np.array_split(t,13):rows+=stream.push(np.full(len(chunk),1100),chunk)
    np.testing.assert_allclose([r["window_end"] for r in rows],[10,12,14])
    assert stream.push([4095],[15])[0]["score"] is None
    assert stream.push(np.full(900,1100),15.01+np.arange(900)/100)==[]
