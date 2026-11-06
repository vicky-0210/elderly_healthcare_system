"""Subject-owned WESAD window packs. Start/end times are seconds since synchronized origin."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from wearable_eda.preprocessing import wrist as _wrist
from wearable_eda.preprocessing.features import window_features
from wearable_eda.preprocessing.wrist import transform_recording


def _signature():
    import inspect
    return hashlib.sha256(b"".join(Path(p).read_bytes() for p in
            (__file__, _wrist.__file__, inspect.getsourcefile(window_features)))).hexdigest()[:12]


def _cache_file(cache, role, identifier):
    h = hashlib.sha256(json.dumps(identifier,sort_keys=True).encode()).hexdigest()[:20]
    folder = cache / "views" / _signature() / role
    folder.mkdir(parents=True,exist_ok=True)
    return folder / (h+".npz")


def _save(path, **arrays):
    temporary = path.with_name(path.stem+"."+uuid.uuid4().hex+".npz")
    np.savez_compressed(temporary, **arrays)
    try:
        os.replace(temporary,path)
    except PermissionError:
        # A concurrent reader can briefly lock the identical completed file on Windows.
        if not path.exists():
            raise
        temporary.unlink(missing_ok=True)


def subject_views(cache, row, **settings):
    cache = Path(cache)
    spec = dict(settings)
    file = _cache_file(cache, "continuous", [row["sha256"], spec])
    if file.exists():
        with np.load(file) as z:
            return z["matrix"],z["channels"].tolist()
    with np.load(cache / f"S{row['subject']}.npz") as z:
        streams = {k:z[k] for k in spec["modalities"]}
        x, channels = transform_recording(streams,rest=z["rest"],**spec)
    _save(file,matrix=x,channels=channels)
    return x, channels


def _eligible(eda):
    bad = ~np.isfinite(eda)
    edges = np.flatnonzero(np.diff(np.r_[False,bad,False])).reshape(-1,2)
    long_gap = np.zeros(len(eda),bool)
    for a,b in edges:
        if b-a>=20:
            long_gap[a:b] = True
    return np.r_[0,np.cumsum(bad)], np.r_[0,np.cumsum(long_gap)]


def build_windows(cache, seconds=60, stride=15, purity=.8, classes=(1,2), calibration="zero",
                  decomposition="mcsaed", modalities=("eda",), filtering="conservative", quality="auto"):
    """Generate within each participant independently. No fitted cross-person transformations."""
    cache = Path(cache)
    manifest = json.loads((cache/"manifest.json").read_text())
    classes = tuple(classes)
    modalities = tuple(modalities)
    if (not set(classes).issubset({1,2,3,4}) or len(set(classes))!=len(classes)
            or len(classes)<2 or not .5<purity<=1 or seconds<2 or stride<=0
            or seconds*4!=int(seconds*4) or stride*4!=int(stride*4)):
        raise ValueError("Invalid classes, purity or 4-Hz window geometry")
    spec = dict(calibration=calibration,decomposition=decomposition,modalities=modalities,
                filtering=filtering,quality=quality)
    hashes = [r["sha256"] for r in manifest["subjects"]]
    file = _cache_file(cache,"windows",[hashes,spec,seconds,stride,purity,classes])
    if file.exists():
        with np.load(file) as z:
            return {k:(z[k].tolist() if k in ("channels","features") else z[k]) for k in z.files}
    windows, feats, targets, groups, starts = [], [], [], [], []
    width, step = int(seconds*4), int(stride*4)
    for row in manifest["subjects"]:
        sid = row["subject"]
        x, channels = subject_views(cache, row, **spec)
        with np.load(cache/f"S{sid}.npz") as z:
            counts = np.vstack((np.zeros((1,8),dtype=np.int64),z["counts"].cumsum(axis=0,dtype=np.int64)))
            end_rest = int(z["rest"][1])
            invalid,long_gap = _eligible(z["eda"])
        # The exact same scoring interval is used for both calibration protocols.
        for a in range(((end_rest+step-1)//step)*step,len(x)-width+1,step):
            b = a+width
            votes = (counts[b]-counts[a])[list(classes)]
            index = int(votes.argmax())
            if (votes[index] < purity*width*175 or invalid[b]-invalid[a]>.2*width
                    or long_gap[b]>long_gap[a]):
                continue
            win = x[a:b].T
            vec, features = window_features(win, channels)
            windows.append(win)
            feats.append(vec)
            groups.append(sid)
            starts.append(a/4)
            targets.append(index)
    if not windows:
        raise ValueError("No eligible windows")
    out = dict(X=np.stack(windows),F=np.stack(feats),y=np.array(targets),groups=np.array(groups),
               start=np.array(starts),end=np.array(starts)+seconds,channels=channels,features=features)
    if not np.isfinite(out["X"]).all() or not np.isfinite(out["F"]).all():
        raise ValueError("Non-finite model input")
    _save(file,**out)
    return out


read_windows = build_windows
