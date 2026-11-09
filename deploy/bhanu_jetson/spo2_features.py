"""Self-contained copies of the SpO2 feature code used for training (numpy + pandas + scipy only, no sliding_window_view so it
also runs on old Jetson Python/numpy). build_deploy.py / replay_test.py check these give the SAME numbers as the training code.
  gbm_features(s)    : GBM window features + causal context features for every 60 s window (hop 10 s) of a 1 Hz SpO2 series
  cnn_channels(s)    : the 6 CNN SpO2 channels (1 Hz): ff, rel120, ok, rel300, rel_mean180, below3_180
NOTE: the training features smooth SpO2 with a centred 10 s filter, i.e. up to 4 s AFTER the window end. The live runner therefore
decides on window [t-65, t-5] at time t (5 s extra latency) so that every value used is real.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import uniform_filter1d

WIN, HOP, MEDIAN = 60, 10, 5


def swv(a, w):
    a = np.ascontiguousarray(a)
    return as_strided(a, shape=(len(a) - w + 1, w), strides=(a.strides[0], a.strides[0]), writeable=False)


def msmooth(v, w=10):
    ok = np.isfinite(v)
    num = uniform_filter1d(np.where(ok, v, 0.0), w, mode="constant")
    den = uniform_filter1d(ok.astype(float), w, mode="constant")
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def trend_feats(v, win, prefix):
    W = swv(v, win)[::HOP]
    S = swv(msmooth(v), win)[::HOP]
    fin = np.isfinite(W)
    cnt = np.maximum(fin.sum(1), 1)
    mean = np.nansum(W, 1) / cnt
    std = np.sqrt(np.maximum(np.nansum(np.where(fin, W - mean[:, None], 0) ** 2, 1) / cnt, 0))
    x = np.arange(win)[None, :].astype(float)
    xm = (x * fin).sum(1) / cnt
    cov = (np.where(fin, (x - xm[:, None]) * (W - mean[:, None]), 0)).sum(1) / cnt
    var = (np.where(fin, (x - xm[:, None]) ** 2, 0)).sum(1) / cnt
    third = win // 3
    f1 = np.nanmean(W[:, :third], 1); f3 = np.nanmean(W[:, -third:], 1)
    mn, mx = np.nanmin(S, 1), np.nanmax(S, 1)
    m = np.where(fin.sum(1) >= 0.5 * win, mean, np.nan)
    rel5 = m - pd.Series(m).rolling(30, min_periods=5).median().shift(1).values
    return pd.DataFrame({f"{prefix}_mean": m, f"{prefix}_std": std, f"{prefix}_min": mn, f"{prefix}_max": mx,
                         f"{prefix}_range": mx - mn, f"{prefix}_slope": cov / np.maximum(var, 1e-9) * 60,
                         f"{prefix}_d13": f3 - f1, f"{prefix}_rel5": rel5})


def spo2_block(s, win=WIN):
    n = len(s)
    st = np.arange(0, n - win + 1, HOP)
    tf = trend_feats(s, win, "spo2")
    ser = pd.Series(pd.Series(s).ffill().values)
    b120 = ser.rolling(120, min_periods=30).quantile(0.9).values
    b60 = ser.rolling(60, min_periods=30).quantile(0.9).values
    prev = lambda b: np.where(st > 0, b[np.maximum(st - 1, 0)], np.nan)
    B120, B60 = prev(b120), prev(b60)
    W = swv(s, win)[::HOP]
    Sm = swv(msmooth(s, 10), win)[::HOP]
    smin = np.nanmin(Sm, 1)
    tf["spo2_drop120"] = B120 - smin
    tf["spo2_drop60"] = B60 - smin
    tf["spo2_dropmean"] = B120 - np.nanmean(W, 1)
    tf["spo2_n3"] = np.nansum(W <= (B120 - 3)[:, None], 1)
    tf["spo2_n4"] = np.nansum(W <= (B120 - 4)[:, None], 1)
    tf["spo2_tmin"] = np.nanargmin(np.where(np.isfinite(Sm), Sm, np.inf), 1) / win
    return tf


def context(df):
    out = {c: df[c].values for c in df.columns}
    for c in df.columns:
        s = df[c]
        out[c + "_m6"] = s.rolling(6, min_periods=1).mean().values
        out[c + "_m30"] = s.rolling(30, min_periods=3).mean().values
        out[c + "_rel"] = (s - s.rolling(180, min_periods=10).median().shift(1)).values
    return pd.DataFrame(out)


def window_keep(ok, win=WIN, max_invalid=0.10):
    """ok: per-second bool (PPG usable). Windows (start every 10 s) with <= 10 % unusable seconds are kept, as in training."""
    st = np.arange(0, len(ok) - win + 1, HOP)
    c = np.concatenate([[0], np.cumsum(~np.asarray(ok, bool))])
    return ((c[st + win] - c[st]) / win) <= max_invalid


def gbm_features(s, ok):
    """Features of the KEPT windows only; the 30-minute context is computed over kept windows (as in training)."""
    return context(spo2_block(s)[window_keep(ok)].reset_index(drop=True))


def causal_spo2(s):
    ff = pd.Series(s).ffill().bfill().values
    base = pd.Series(ff).rolling(120, min_periods=30).quantile(0.9).shift(1).bfill().values
    return ff, ff - base


def cnn_channels(s):
    """s: 1 Hz SpO2 with NaN for missing/invalid -> array (6, n)."""
    sm = pd.Series(s).rolling(MEDIAN, min_periods=1).median().values
    ff, rel = causal_spo2(sm)
    r = pd.Series(rel)
    base300 = pd.Series(ff).rolling(300, min_periods=30).quantile(0.9).shift(1).bfill().values
    ok = np.isfinite(s).astype(float)
    return np.stack([ff, rel, ok, ff - base300, r.rolling(180, min_periods=1).mean().values,
                     (r < -3).astype(float).rolling(180, min_periods=1).mean().values])
