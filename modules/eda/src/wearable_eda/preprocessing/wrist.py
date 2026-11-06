"""4 Hz wrist preprocessing. All times refer to completed quarter-second bins."""
import numpy as np
import pandas as pd
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

FS = 4
CORE = ("raw", "clean", "tonic", "phasic", "relative_tonic", "relative_phasic",
        "tonic_slope", "phasic_slope", "phasic_energy", "quality", "artifact", "missing")
MOTION = ("acc_magnitude", "acc_dynamic", "acc_rms", "acc_jerk", "motion")
BVP = ("bvp_rms", "bvp_rate", "bvp_pulse")


def carry(values, fallback=0.0):
    arr = np.asarray(values, dtype=np.float64)
    good = np.isfinite(arr)
    indices = np.maximum.accumulate(np.where(good, np.arange(len(arr)), -1))
    return np.where(indices >= 0, arr[np.maximum(indices, 0)], fallback)


def mean_past(x, n):
    c = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    j = np.arange(1, len(c))
    k = np.maximum(j-n, 0)
    return (c[j]-c[k]) / (j-k)


def lowpass(x, hz, fs=FS):
    sos = signal.butter(2, hz, fs=fs, btype="low", output="sos")
    return signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos)*x[0])[0]


def motion_views(acc):
    a = np.asarray(acc, dtype=np.float64) / 64.0
    if a.ndim != 2 or a.shape[1] != 3 or len(a) % 8:
        raise ValueError("Expected 32 Hz ACC with three axes and eight samples per EDA tick")
    a = np.column_stack([carry(v) for v in a.T])
    gravity = np.column_stack([lowpass(v, .25, 32) for v in a.T])
    mag = np.linalg.norm(a, axis=1)
    dyn = np.linalg.norm(a-gravity, axis=1)
    jerk = np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=1)*32
    # Each output uses the preceding one-second interval ending at this EDA tick.
    avg = lambda x: mean_past(x, 32)[7::8]
    std = np.sqrt(np.maximum(0, avg(mag*mag)-avg(mag)**2))
    rms = np.sqrt(avg(dyn**2))
    intensity = np.clip((rms + std)/.5, 0, 1)
    return dict(zip(MOTION, (avg(mag), avg(dyn), rms, avg(jerk), intensity)))


def bvp_views(bvp):
    b = carry(np.asarray(bvp).ravel())
    sos = signal.butter(2, [.6, 3.5], btype="bandpass", fs=64, output="sos")
    v = signal.sosfilt(sos, b, zi=signal.sosfilt_zi(sos)*b[0])[0]
    rms = np.sqrt(np.maximum(mean_past(v*v, 128), 0))
    # Confirm a peak with the following native sample; emit only at confirmation.
    candidate = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
                               & (v[1:-1] > .5*rms[1:-1])) + 2
    impulses = np.zeros(len(b))
    last = -1000
    for j in candidate:
        if j-last >= 18:
            impulses[j] = 1
            last = j
    pulse = impulses.reshape(-1, 16).sum(axis=1)
    return dict(zip(BVP, (rms[15::16], mean_past(pulse, 48)*240, pulse)))


def assess(eda, motion, quality):
    raw = np.asarray(eda, dtype=np.float64).ravel()
    missing = ~np.isfinite(raw)
    physical = (raw < 0) | (raw > 100)
    x = carry(np.where(physical, np.nan, raw))
    delta = np.diff(x, prepend=x[0])
    local = pd.Series(np.abs(delta)).rolling(121, min_periods=1).median().shift(1).fillna(0).to_numpy()
    deviation = np.abs(np.abs(delta)-local)
    mad = pd.Series(deviation).rolling(121, min_periods=1).median().shift(1).fillna(0).to_numpy()
    jump = np.clip((np.abs(delta)-local-8*1.4826*mad-.08)/.5, 0, 1)
    unchanged = np.abs(delta) < 1e-8
    age = np.arange(len(x)) - np.maximum.accumulate(np.where(~unchanged, np.arange(len(x)), 0))
    flat = (age >= 80) & unchanged
    reversed_step = (delta * np.r_[0., delta[:-1]] < 0)
    # Isolated positive SCRs and movement alone cannot trigger strong correction.
    conf = jump * (.15+.35*(delta<0)+.25*reversed_step)
    if quality == "motion":
        conf = np.maximum(conf, jump*(.15+.6*motion))
    conf = np.maximum(conf, np.where(flat, .65, 0.))
    conf = np.maximum(conf, np.where((flat & (x < .02)) | physical | missing, 1., 0.))
    if quality == "off":
        conf[:] = 0.
    previous = np.r_[x[0], x[:-1]]
    clean = x - conf*np.clip(conf-.4, 0, .6)/.6*(x-previous)
    return x, clean, conf, missing.astype(float)


def adaptive_components(x, confidence):
    tonic = np.empty_like(x)
    tonic[0] = x[0]
    for i in range(1, len(x)):
        diff = x[i]-tonic[i-1]
        # Slower upward tracking preserves excitatory SCRs; downward tracking
        # follows relaxation. Reliability damps tonic updates on suspect steps.
        rate = .006 if diff > 0 else .095
        tonic[i] = tonic[i-1] + rate*(1-.9*confidence[i])*diff
    excess = np.maximum(x-tonic, 0.)
    phasic = signal.lfilter([.6], [1., -.4], excess)
    return tonic, phasic


def decompose(x, conf, method):
    if method == "mcsaed":
        return adaptive_components(x, conf)
    if method == "none":
        return x.copy(), np.zeros_like(x)
    if method == "sparse":
        # A causal one-sided L1 innovation model with exponential SCR decay.
        # It is a lightweight sparse comparator, distinct from the sparsEDA package.
        baseline = lowpass(x, .035)
        residual = np.maximum(x-baseline, 0)
        decay = np.exp(-.25/2)
        driver = np.maximum(residual-decay*np.r_[0.,residual[:-1]]-.008, 0)
        return baseline, signal.lfilter([1.], [1., -decay], driver)
    if method == "choi":
        # Eq. 1, Choi et al. 2012. Solve the pentadiagonal linear system.
        d = sparse.diags((np.ones(len(x)-2),-2*np.ones(len(x)-2),np.ones(len(x)-2)),
                         (0,1,2),shape=(len(x)-2,len(x)),format="csc")
        tonic = spsolve(sparse.eye(len(x),format="csc")+1500.**2*(d.T@d), x)
        return tonic, x-tonic
    if method == "cvxeda":
        from cvxeda import cvxEDA
        tonic = np.empty_like(x)
        phasic = np.empty_like(x)
        # Overlapping padded blocks bound solver memory; crop context. This is
        # a retrospective comparator and must never be exported for online use.
        for start in range(0, len(x), 480):
            end = min(len(x), start+480)
            left, right = max(0,start-120), min(len(x),end+120)
            result = tuple(cvxEDA(x[left:right], .25, options={"show_progress":False,
                                                          "reltol":1e-7,"maxiters":100}))
            phasic[start:end] = np.asarray(result[0]).ravel()[start-left:end-left]
            tonic[start:end] = np.asarray(result[2]).ravel()[start-left:end-left]
        if not np.isfinite(tonic+phasic).all():
            raise ArithmeticError("cvxEDA returned non-finite components")
        return tonic, phasic
    raise ValueError(f"Unknown decomposition {method}")


def _continuous_block(streams, modalities, filtering, quality, decomposition):
    if "eda" not in modalities or set(modalities)-{"eda","acc","temp","bvp"}:
        raise ValueError("EDA is required and only wrist EDA, ACC, TEMP, BVP are supported")
    if quality == "auto":
        quality = "motion" if "acc" in modalities else "eda"
    if quality not in ("motion","eda","off") or (quality=="motion" and "acc" not in modalities):
        raise ValueError("Motion quality requires an enabled ACC stream")
    n = len(streams["eda"])
    for key, rate in (("acc",8),("temp",1),("bvp",16)):
        if key in modalities and len(streams[key]) != n*rate:
            raise ValueError(f"Unequal synchronized duration for {key}")
    extra = motion_views(streams["acc"]) if "acc" in modalities else {}
    x, repaired, artifact, missing = assess(streams["eda"], extra.get("motion",0.), quality)
    if filtering not in ("conservative","1hz","raw"):
        raise ValueError(f"Unknown filter {filtering}")
    clean = repaired if filtering == "raw" else lowpass(repaired, .65 if filtering=="conservative" else 1.)
    tonic, phasic = decompose(clean, artifact, decomposition)
    zeros = np.zeros(n)
    views = dict(zip(CORE, (x,clean,tonic,phasic,zeros,zeros,np.diff(tonic,prepend=tonic[0])*4,
                            np.diff(phasic,prepend=phasic[0])*4,mean_past(phasic**2,16),
                            1-artifact,artifact,missing)))
    views.update(extra)
    if "temp" in modalities:
        views["temperature"] = carry(streams["temp"])
    if "bvp" in modalities:
        views.update(bvp_views(streams["bvp"]))
    return np.column_stack(tuple(views.values())).astype("f4"), list(views)


def continuous(streams, modalities, filtering, quality, decomposition):
    """Start fresh physiological filter states after an EDA gap of five seconds."""
    if "eda" not in streams or np.ndim(streams["eda"]) != 1 or len(streams["eda"]) == 0:
        raise ValueError("Expected a nonempty 1D wrist EDA stream")
    missing = ~np.isfinite(streams["eda"])
    edges = np.flatnonzero(np.diff(np.r_[False, missing, False])).reshape(-1, 2)
    gaps = [(a,b) for a,b in edges if b-a >= 20]
    if not gaps:
        return _continuous_block(streams, modalities, filtering, quality, decomposition)
    # Raw timestamps that disappear altogether are rejected by the acquisition
    # contract; explicit missing bins stay on the original timeline.
    spans = [0] + [int(edge) for gap in gaps for edge in gap] + [len(missing)]
    chunks = []
    valid_names = list(CORE)
    if "acc" in modalities:
        valid_names.extend(MOTION)
    if "temp" in modalities:
        valid_names.append("temperature")
    if "bvp" in modalities:
        valid_names.extend(BVP)
    for left,right in zip(spans,spans[1:]):
        if right <= left:
            continue
        if missing[left:right].all():
            chunk = np.zeros((right-left,len(valid_names)),dtype="f4")
            chunk[:,valid_names.index("missing")] = 1
            chunk[:,valid_names.index("artifact")] = 1
        else:
            part = {k:streams[k][left*m:right*m] for k,m in
                    (("eda",1),("acc",8),("temp",1),("bvp",16)) if k in modalities}
            if decomposition in ("choi","cvxeda") and right-left<40:
                raise ValueError("Retrospective decomposition needs at least 10s of continuous signal")
            chunk, names = _continuous_block(part,modalities,filtering,quality,decomposition)
            if names != valid_names:
                raise ValueError("Inconsistent channel order across a gap")
        chunks.append(chunk)
    return np.concatenate(chunks), valid_names


def calibrate(values, channels, calibration, reference=None):
    if calibration in ("absolute","zero"):
        return values
    output = values.copy()
    for source, target, floor in (("tonic","relative_tonic",.05),("phasic","relative_phasic",.02)):
        x = values[:,channels.index(source)]
        if calibration == "rest":
            if reference is None or len(reference)<480:
                raise ValueError("Rest calibration requires at least 120 seconds of designated resting data")
            ok = (reference[:,channels.index("quality")]>.4) & (reference[:,channels.index("missing")]==0)
            if ok.sum()<384:
                raise ValueError("Rest calibration has insufficient usable EDA")
            xb = reference[ok,channels.index(source)]
            center = np.median(xb)
            scale = max(1.4826*np.median(np.abs(xb-center)),floor)
        elif calibration == "rolling":
            # Prospective deployment sensitivity only, outside formal LOSO.
            center = pd.Series(x).rolling(480,min_periods=1).median().shift(1).fillna(x[0]).to_numpy()
            scale = np.maximum(floor,1.4826*pd.Series(abs(x-center)).rolling(480,min_periods=1).median().shift(1).fillna(0).to_numpy())
        else:
            raise ValueError(f"Unknown calibration {calibration}")
        output[:,channels.index(target)] = (x-center)/scale
    return output


def transform_recording(streams, *, rest=None, calibration="zero", decomposition="mcsaed",
                        modalities=("eda",), filtering="conservative", quality="auto"):
    """Transform a uniform-rate synchronized prefix. No protocol labels accepted.

    rest is an explicit half-open interval in 4 Hz ticks, designated by the
    operator. Use predictions only after its end. Offline decomposers require
    future observations and are ineligible for the deployment path.
    """
    values, names = continuous(streams,modalities,filtering,quality,decomposition)
    reference = None
    if calibration == "rest":
        if rest is None or len(rest)!=2 or not (0<=rest[0]<rest[1]<=len(values)):
            raise ValueError("Missing or invalid explicit rest interval")
        # Fit the reference exclusively inside the explicit 120s resting capture.
        # Restart filters, including offline solvers, at those interval endpoints.
        segment = {k:streams[k][rest[0]*m:rest[1]*m] for k,m in
                   (("eda",1),("acc",8),("temp",1),("bvp",16)) if k in modalities}
        reference, _ = continuous(segment,modalities,filtering,quality,decomposition)
    calibrated = calibrate(values,names,calibration,reference)
    if calibration == "rest":
        for name in ("relative_tonic", "relative_phasic"):
            calibrated[:rest[1],names.index(name)] = 0.
    return calibrated, names


