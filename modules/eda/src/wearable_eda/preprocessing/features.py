"""Hand-engineered wrist physiology summaries shared by training and replay."""
import numpy as np
from scipy.signal import find_peaks, peak_widths


def window_features(X, channels, *, prominence=.03):
    """Aggregate one [channel,time] window in physical units; no labels required."""
    v = dict(zip(channels,X))
    t, p = v["tonic"], v["phasic"]
    length = len(t)
    u = np.arange(length, dtype=float)/4
    u -= u.mean()
    slope = float(u@t / (u@u))
    peaks, props = find_peaks(p, prominence=prominence, distance=4)
    if len(peaks):
        amps = props["prominences"]
        widths, _, left, right = peak_widths(p, peaks, rel_height=.5,
                            prominence_data=(amps,props["left_bases"],props["right_bases"]))
        timing = (amps.mean(),amps.max(),widths.mean()/4,
                  (peaks-left).mean()/4,(right-peaks).mean()/4)
    else:
        timing = (0.,)*5
    f = {"scl_mean":t.mean(), "scl_median":np.median(t), "scl_sd":t.std(),
         "scl_slope":slope, "scl_range":np.ptp(t),
         "scl_quartile_change":t[-length//4:].mean()-t[:length//4].mean(),
         "scr_count":len(peaks), "scr_amp_mean":timing[0], "scr_amp_max":timing[1],
         "scr_halfwidth_s":timing[2], "scr_halfrise_s":timing[3], "scr_halfrecovery_s":timing[4],
         "scr_energy":np.mean(p*p), "scr_positive_area":np.sum(np.maximum(p,0))/4,
         "relative_scl_mean":v["relative_tonic"].mean(),
         "relative_scr_mean":v["relative_phasic"].mean(),
         "artifact_ratio":np.mean(v["artifact"]>=.5), "quality_mean":v["quality"].mean(),
         "missing_ratio":v["missing"].mean(),
         "eda_abs_change":np.abs(np.diff(v["clean"])).mean()}
    for name in ("acc_magnitude","acc_dynamic","acc_rms","acc_jerk","motion",
                 "temperature","bvp_rms","bvp_rate","bvp_pulse"):
        if name in v:
            a = v[name]
            f[name+"_mean"] = a.mean()
            f[name+"_sd"] = a.std()
            if name=="temperature":
                f["temperature_slope"] = float(u@a / (u@u))
    return np.array(list(f.values()), dtype="f4"), list(f)
