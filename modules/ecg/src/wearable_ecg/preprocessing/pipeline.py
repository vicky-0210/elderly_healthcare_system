"""Conservative window preprocessing in physical mV.

The protected regions and fiducials are engineering proxies, not a clinical
delineator. Their limitations and the controller's synthetic-only validation
are part of the saved experiment report. No fitted quantities use folds 9/10.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pywt
from scipy import signal, interpolate, ndimage

FS = 360
FEATURE_NAMES = ["rms_mv", "span_mv", "diff_noise_mv", "mains_fraction", "low_fraction",
                 "flat_fraction", "peak_count", "rr_cv"]
VOLTAGE_NAMES = ["p01_mv", "p99_mv", "rms_mv", "span_mv", "r_amplitude_mv", "s_amplitude_mv",
                 "qrs_area_mv_s", "qrs_width_proxy_s", "st_proxy_mv", "t_proxy_mv",
                 "heart_rate_proxy", "rr_cv", "correction_rms_mv", "disagreement_rms_mv",
                 "quality", "delineation_available"]


@dataclass(frozen=True)
class Setting:
    name: str
    baseline: bool = False
    shrink: float = 0.0
    mains: bool = False


SETTINGS = [Setting("minimal"), Setting("baseline", True),
            Setting("bounded_swt_light", True, .20, True),
            Setting("bounded_swt_medium", True, .45, True)]


def resample_mv(wave, source_hz=500, target_hz=360):
    from math import gcd
    g = gcd(source_hz, target_hz)
    # Retain calibration precision through filtering; narrow only at model input.
    return signal.resample_poly(np.asarray(wave,dtype=np.float64), target_hz // g, source_hz // g, axis=0).astype(np.float32)


def landmarks(x, fs=FS):
    # Detect on an auxiliary copy; never feed this bandpass into the voltage path.
    z = signal.sosfiltfilt(signal.butter(2, [5, 25], fs=fs, btype="bandpass", output="sos"), x)
    envelope = ndimage.uniform_filter1d(z * z, max(3, int(.075 * fs)))
    peaks, _ = signal.find_peaks(envelope, distance=int(.25 * fs),
                                 prominence=max(np.quantile(envelope, .80) * .25, 1e-7))
    aligned = []
    for p in peaks:
        a, b = max(0, p - int(.065 * fs)), min(len(x), p + int(.065 * fs) + 1)
        aligned.append(a + np.argmax(np.abs(z[a:b])))
    return np.array(sorted(set(p for p in aligned if .25 * fs < p < len(x) - .45 * fs)), dtype=int)


def quality_features(x, peaks=None, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    peaks = landmarks(x, fs) if peaks is None else peaks
    f, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    total = psd.sum() + 1e-12
    rr = np.diff(peaks) / fs
    return np.asarray([np.std(x), np.quantile(x, .99)-np.quantile(x, .01),
                       np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / .6745,
                       psd[(abs(f-50)<1.5) | (abs(f-60)<1.5)].sum()/total,
                       psd[f < .7].sum()/total, np.mean(np.abs(np.diff(x)) < 1e-7),
                       len(peaks), np.std(rr)/(np.mean(rr)+1e-9) if len(rr)>1 else 0], dtype=np.float32)


def reliability(x, feats, fs=FS):
    # No penalty for rhythm irregularity or high voltage alone.
    diff = np.abs(np.diff(x, prepend=x[0]))
    local = ndimage.uniform_filter1d(diff, max(3, int(.15*fs)))
    q = np.exp(-np.maximum(local - .10, 0) / .25)
    q *= np.clip(1 - feats[5], .05, 1)
    return np.clip(q, .05, 1).astype(np.float32)


def robust_baseline(x, peaks, fs=FS):
    if len(peaks) < 3: return np.full_like(x, np.median(x))
    # Use candidate pre-QRS (PR) samples. Short RR or absent safe intervals are
    # excluded; true isoelectric status is not guaranteed without delineation.
    times, values = [], []
    for i, p in enumerate(peaks):
        a, b = p-int(.22*fs), p-int(.12*fs)
        if a < 0 or (i and a < peaks[i-1]+int(.40*fs)): continue
        times.append((a+b)/2/fs)
        values.append(np.median(x[a:b]))
    if len(times) < 4: return np.full_like(x, np.median(x))
    t, v = np.asarray(times), np.asarray(values)
    weights = np.ones(len(t))
    smoothing = len(t) * .025**2
    for _ in range(3):
        fit = interpolate.UnivariateSpline(t, v, w=weights, k=min(3, len(t)-1), s=smoothing, ext=3)
        residual = v - fit(t)
        scale = max(np.median(np.abs(residual))/.6745, .01)
        weights = np.minimum(1, 1.5*scale/(np.abs(residual)+1e-10))
    return fit(np.arange(len(x))/fs)


def correct(x, index, peaks=None, fs=FS):
    cfg = SETTINGS[int(index)]
    x = np.asarray(x, dtype=np.float64)
    if index == 0: return x.astype(np.float32)
    peaks = landmarks(x, fs) if peaks is None else peaks
    y = x - robust_baseline(x, peaks, fs) if cfg.baseline else x.copy()
    if cfg.mains:
        f, psd = signal.welch(y, fs=fs, nperseg=min(len(y), fs*4))
        for hz in (50, 60):
            band = abs(f-hz) <= .5
            neighbors = (abs(f-hz)>1) & (abs(f-hz)<4)
            if psd[band].mean() > 8*(psd[neighbors].mean()+1e-12):
                t = np.arange(len(y))/fs
                design = np.stack([np.sin(2*np.pi*hz*t), np.cos(2*np.pi*hz*t)], axis=1)
                y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if cfg.shrink:
        pad = (-len(y)) % 8
        yp = np.pad(y, (0,pad), mode="reflect") if pad else y
        protected = np.ones(len(yp))
        for p in peaks:
            protected[max(0,p-int(.1*fs)):min(len(yp),p+int(.40*fs))] = .20
        coeffs = pywt.swt(yp, "sym4", level=3)
        processed = []
        for level, (a,d) in enumerate(coeffs):
            sigma = np.median(np.abs(d-np.median(d)))/.6745
            # Only the two highest bands; bounded subtraction, never zero bands.
            strength = cfg.shrink if level >= 1 else 0
            threshold = strength * sigma * protected
            processed.append((a, np.sign(d)*np.maximum(np.abs(d)-threshold,0)))
        y = pywt.iswt(processed, "sym4")[:len(x)]
    return np.asarray(y, dtype=np.float32)


def morphology(x, peaks, fs=FS):
    values = []
    for p in peaks:
        if p-int(.22*fs)<0 or p+int(.42*fs)>=len(x): continue
        baseline = np.median(x[p-int(.22*fs):p-int(.12*fs)])
        qrs = x[p-int(.08*fs):p+int(.10*fs)]-baseline
        r, s = np.max(qrs), np.min(qrs)
        width = np.sum(np.abs(qrs) > .15*max(abs(r),abs(s),1e-5))/fs
        st = np.mean(x[p+int(.12*fs):p+int(.16*fs)])-baseline
        tw = x[p+int(.18*fs):p+int(.40*fs)]-baseline
        values.append([r,s,np.sum(abs(qrs))/fs,width,st,tw[np.argmax(abs(tw))]])
    return np.median(values, axis=0) if values else np.zeros(6)


def voltage_features(x, original, disagreement, peaks, q, fs=FS):
    rr = np.diff(peaks)/fs
    v = [np.quantile(x,.01), np.quantile(x,.99), np.std(x), np.ptp(x)]
    v.extend(morphology(x,peaks,fs))
    v.extend([60/(np.median(rr)+1e-9)/100 if len(rr) else 0,
              np.std(rr)/(np.mean(rr)+1e-9) if len(rr)>1 else 0,
              np.sqrt(np.mean((x-original)**2)), disagreement, np.mean(q), float(len(peaks)>=3)])
    return np.asarray(v,dtype=np.float32)


def process(x, controller=None, variant="constrained", fs=FS):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or len(x) != fs*10 or not np.isfinite(x).all():
        raise ValueError("Require a finite, calibrated, single-lead 10-second window")
    peaks = landmarks(x, fs)
    feats = quality_features(x, peaks, fs)
    chosen = 0
    if variant == "constrained" and controller is not None:
        chosen = int(controller.predict(feats[None])[0])
    elif variant == "swt": chosen = 3
    if variant in {"conventional_fixed", "conventional_zscore"}:
        y = signal.sosfiltfilt(signal.butter(3,[.5,100],fs=fs,btype="bandpass",output="sos"),x).astype(np.float32)
    else: y = correct(x,chosen,peaks,fs)
    nearby = correct(x, max(0, chosen-1), peaks, fs) if chosen else x
    disagreement = float(np.sqrt(np.mean((y-nearby)**2)))
    q = reliability(x,feats,fs)
    v = voltage_features(y,x,disagreement,peaks,q,fs)
    # Keep physical amplitude unchanged. The model creates its own shape view.
    if variant == "conventional_zscore":
        y = (y-np.mean(y))/max(float(np.std(y)),1e-4)
        v[:14] = 0  # Explicit amplitude-removal ablation.
    return y.astype(np.float32), v, q, chosen


def corrupt(x, rng, severity=1.0, fs=FS):
    t = np.arange(len(x))/fs
    noise = (.03*severity*np.sin(2*np.pi*rng.uniform(.12,.40)*t+rng.uniform(0,6.28))
             + .008*severity*np.sin(2*np.pi*50*t+rng.uniform(0,6.28))
             + rng.normal(0,.008*severity,len(x)))
    return (x+noise).astype(np.float32)


def fit_controller(waves, labels, output, seed=42, n_per_class=128):
    """Train-only cost selection on synthetic pairs; conservative hard budgets.

    Individual pairs must satisfy amplitude/width/ST proxy limits, and the
    clean signal must also be preserved. If none passes, select identity.
    This is a pilot realization of the protocol, not validated morphology.
    """
    from pathlib import Path
    import joblib
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from wearable_ecg.runtime import save_json
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels==c),min(n_per_class,np.sum(labels==c)),replace=False) for c in (0,1)])
    rows, features, targets = [], [], []
    for idx in ids:
        clean = waves[idx]
        peaks = landmarks(clean)
        if len(peaks)<3: continue
        ref = morphology(clean,peaks)
        for strength in (.0,1.,3.):
            noisy = corrupt(clean,rng,strength)
            best, best_error = 0, float(np.mean(abs(noisy-clean)))
            for j in range(1,len(SETTINGS)):
                result = correct(noisy,j,peaks)
                phi = morphology(result,peaks)
                clean_result = correct(clean,j,peaks)
                amplitude = np.max(abs(phi[:2]-ref[:2])/np.maximum(abs(ref[:2]),.2))
                width = abs(phi[3]-ref[3])
                st = abs(phi[4]-ref[4])
                error = float(np.mean(abs(result-clean)))
                clean_harm = float(np.mean(abs(clean_result-clean)))
                feasible = amplitude<=.05 and width<=.010 and st<=.020 and clean_harm<=.025
                if feasible and error<best_error: best,best_error=j,error
                rows.append({"sample_index":int(idx),"noise_strength":strength,"setting":j,
                             "mae_mv":error,"amplitude_relative_error":float(amplitude),
                             "width_proxy_error_ms":float(width*1000),"st_proxy_error_uv":float(st*1000),
                             "clean_harm_mae_mv":clean_harm,"feasible":bool(feasible)})
            features.append(quality_features(noisy))
            targets.append(best)
    if not features: raise ValueError("No usable training references for preprocessing controller")
    model = DecisionTreeClassifier(max_depth=4,min_samples_leaf=24,random_state=seed)
    model.fit(features,targets)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output/"controller.joblib")
    pd.DataFrame(rows).to_csv(output/"controller_training_pairs.csv",index=False)
    save_json(output/"controller.json", {"fit_folds":[1,2,3,4,5,6],"seed":seed,
        "reference_count":len(set(r["sample_index"] for r in rows)),"pair_count":len(targets),
        "setting_counts":{SETTINGS[k].name:int(np.sum(np.array(targets)==k)) for k in range(len(SETTINGS))},
        "settings":[asdict(s) for s in SETTINGS],"features":FEATURE_NAMES,
        "limitation":"Automatic fiducial proxies, synthetic artifacts and reference-quality (not clean ground truth) ECGs; requires manual landmark review and real-noise validation."})
    return model
