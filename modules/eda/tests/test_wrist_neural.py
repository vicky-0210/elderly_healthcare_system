"""Regression coverage for subject partitions, wrist models and checkpoint inference."""
import numpy as np
import pytest
import torch

from wearable_eda.inference.neural import NeuralPredictor, scale
from wearable_eda.models.wrist import WristNet
from wearable_eda.training.neural_experiments import catalogue, selected_channels
from wearable_eda.training.neural_suite import pick_subjects, prepare, save_checkpoint


CHANNELS = ["raw","clean","tonic","phasic","relative_tonic","relative_phasic","tonic_slope",
            "phasic_slope","phasic_energy","quality","artifact","missing","acc_magnitude",
            "acc_dynamic","acc_rms","acc_jerk","motion","temperature","bvp_rms","bvp_rate","bvp_pulse"]


def fixture():
    rng = np.random.default_rng(128)
    groups = np.repeat([2,3,4,5,6],3)
    x = rng.normal(size=(len(groups),len(CHANNELS),240)).astype("f")
    for name in ("quality","artifact","missing"):
        x[:,CHANNELS.index(name)] = rng.uniform(size=x[:,0].shape)
    return dict(X=x,F=rng.normal(size=(len(groups),4)).astype("f"),channels=CHANNELS,
                features=["a","b","c","d"],groups=groups,y=np.arange(len(groups))%2)


def test_partitions_and_frozen_scaling():
    d = fixture()
    a,v,r = pick_subjects(d["groups"],2)
    assert 2 not in set(a)|set(v)|set(r)
    assert set(a).isdisjoint(v) and set(r)==set(a)|set(v)
    scale1 = prepare(d,d["X"],a,CHANNELS,"cpu")["scaler"]
    d["X"][~np.isin(d["groups"],a)] += 200
    d["F"][~np.isin(d["groups"],a)] -= 500
    scale2 = prepare(d,d["X"],a,CHANNELS,"cpu")["scaler"]
    for key in scale1:
        np.testing.assert_equal(scale1[key],scale2[key])


def test_graphs_and_reload(tmp_path):
    torch.set_num_threads(1)
    d = fixture()
    for spec in catalogue().values():
        names = selected_channels(CHANNELS,spec)
        idx = [CHANNELS.index(n) for n in names]
        pack = prepare(d,d["X"][:,idx],np.unique(d["groups"])[:-1],names,"cpu")
        model = WristNet(spec,names,len(d["features"]),len(spec["classes"]),
                         pack["scaler"]["xc"],pack["scaler"]["xs"])
        logits, reg = model(pack["x"][:3],pack["f"][:3])
        assert logits.shape==(3,len(spec["classes"])) and torch.isfinite(logits).all()
        (logits.square().mean()+reg).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    spec = catalogue()["pg_no_components_no_hand_zero"]
    names = selected_channels(CHANNELS,spec)
    idx = [CHANNELS.index(n) for n in names]
    pack = prepare(d,d["X"][:,idx],np.unique(d["groups"])[:-1],names,"cpu")
    model = WristNet(spec,names,len(d["features"]),2,pack["scaler"]["xc"],pack["scaler"]["xs"]).eval()
    file = tmp_path/"model.pt"
    save_checkpoint(file,spec,names,d["features"],model,pack,.87)
    loaded = NeuralPredictor(file)
    with torch.inference_mode():
        expected = (model(pack["x"],pack["f"])[0]/.87).softmax(1).numpy()
    np.testing.assert_allclose(loaded.predict_windows(d["X"],d["F"],CHANNELS,d["features"]),expected,rtol=1e-5)
    with pytest.raises(ValueError):
        loaded.predict_windows(d["X"][:,:,:-4],d["F"],CHANNELS,d["features"])
