import json

import numpy as np
from scipy.special import expit

from wearable_eda.inference.linear import LinearPredictor
from wearable_eda.inference.predictor import EDAPredictor


def test_linear_temperature_and_imputation(tmp_path):
    path = tmp_path/"linear.json"
    path.write_text(json.dumps(dict(features=["a","b"],impute=[1,2],center=[1,0],
                               scale=[2,4],weights=[.8,-.6],intercept=.4,temperature=2.)))
    predictor = LinearPredictor(path)
    result = predictor.predict_windows([[3,np.nan]],["a","b"])
    expected = expit((.8-.3+.4)/2)
    np.testing.assert_allclose(result,[[1-expected,expected]])


def test_rest_must_be_complete_and_disjoint():
    model = EDAPredictor()
    model.contract = dict(family="neural",experiment="synthetic",format="wesad-wrist-v1",
                         modalities=["eda"],seconds=60,calibration="rest",decomposition="mcsaed",
                         quality="auto",filtering="conservative")

    class FixedScores:
        def predict_windows(self, x, f, channels, features):
            return np.array([[.75,.25]])

    model.model = FixedScores()
    t = np.arange(180*4)/4
    payload = dict(clock_domain="device",stream_id="test",
                streams={"eda":dict(samples=1+.15*np.sin(t),timestamps=t,unit="uS")})
    assert model.predict(payload)["scores"] is None
    assert model.predict(dict(payload,rest_interval_s=[1,121]))["status"]=="insufficient_signal"
    assert model.predict(dict(payload,rest_interval_s=[0,120]))["status"]=="ok"
