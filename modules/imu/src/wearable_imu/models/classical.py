"""Compact statistical descriptors, shared by research and on-device prediction."""
import numpy as np


def descriptors(windows):
    # B x time x channel. Include contrasts across halves without a trial clock.
    half=windows.shape[1]//2
    return np.concatenate((windows.mean(1),windows.std(1),windows.min(1),windows.max(1),
        np.sqrt(np.mean(windows**2,axis=1)),np.ptp(windows,axis=1),
        windows[:,half:].mean(1)-windows[:,:half].mean(1)),axis=1).astype("f")


def logits(model,x):
    if hasattr(model,"decision_function"): return model.decision_function(x).astype("f")
    p=np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
    return np.log(p/(1-p)).astype("f")
