"""Read-only projection of the published OSASUD pandas protocol-4 archive."""
import hashlib
import json
import mmap
from pathlib import Path
import pickletools
import struct
import numpy as np
import pandas as pd

CACHE=Path("data/ppg/processed/osasud_v1")
RESULT=Path("research/results/osasud_v1")
RUN=Path("artifacts/ppg/runs/osasud_v1")
LABELS=["NONE","HYPOPNEA","APNEA-CENTRAL","APNEA-OBSTRUCTIVE","APNEA-MIXED"]
N=961357
ARCHIVE_MD5="b9ed560f4abf6100ce02cfc7cb0c121f"
ARCHIVE_SHA256="dbfac3e44ab7dcc0b05e68657b7dd90b21b8c47e8a23a8378bc10de9b81a343d"


def project(source):
    # The archive's object block contains millions of NumPy arrays. Decode only
    # the first three object columns (patient, event, pleth), without executing pickle.
    with source.open("rb") as f:
        fingerprint=hashlib.file_digest(f,"sha256").hexdigest()
        if fingerprint!=ARCHIVE_SHA256: raise ValueError("Unrecognized OSASUD archive; verify release layout before decoding")
        f.seek(0)
        ids=np.empty(N,dtype="i2"); labels=np.empty(N,dtype="i1")
        signal=np.full((N,80),np.nan,dtype="f")
        strings={}; memo=0; objects=0; started=False; wave=0; previous=None
        dtype=None; defining=None; dtype_references={}; new_dtype=False
        for op,arg,_ in pickletools.genops(f):
            name=op.name
            if name=="MEMOIZE":
                if previous=="SHORT_BINUNICODE": strings[memo]=value
                if new_dtype: dtype_references[memo]=defining; new_dtype=False
                memo+=1
            elif objects<2*N:
                if name=="SHORT_BINUNICODE":
                    value=arg
                    if not started and arg=="1": started=True
                    item=arg
                elif name in ("BINGET","LONG_BINGET") and started: item=strings[arg]
                elif started and name in ("NONE","BINFLOAT"): item=None
                else: item=False
                if started and item is not False:
                    if objects<N: ids[objects]=int(item)
                    else: labels[objects-N]=LABELS.index(item) if item in LABELS else -1
                    objects+=1
            elif name=="SHORT_BINUNICODE":
                value=arg
                if arg in ("f8","i8"):
                    dtype=np.dtype("<"+arg); defining=dtype
            elif name=="REDUCE" and defining is not None and previous=="MEMOIZE": new_dtype=True
            elif name=="BUILD": defining=None
            elif name in ("BINGET","LONG_BINGET") and arg in dtype_references: dtype=dtype_references[arg]
            elif name in ("BINBYTES","SHORT_BINBYTES") and len(arg)>=8:
                if dtype is None: raise ValueError("Undecoded array dtype")
                if len(arg)==80*dtype.itemsize: signal[wave]=np.frombuffer(arg,dtype=dtype)
                else: raise ValueError(f"Unexpected waveform width at source row {wave}")
                wave+=1
                if wave==N: break
            previous=name
        if wave!=N or objects!=2*N: raise ValueError("Incomplete projection")
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
            marker=b"B"+struct.pack("<I",8*N)
            offset=data.find(marker,data.rfind(b"pandas.core.arrays.datetimes"))
            if offset<0 or b"M8" not in data[offset-150:offset]: raise ValueError("Datetime block missing")
            time=np.frombuffer(data,dtype="<i8",count=N,offset=offset+5).copy()
    if np.any(time%1_000_000_000): raise ValueError("Unexpected timestamp resolution")
    return ids,time//1_000_000_000,labels,signal


def prepare(source):
    source=Path(source).resolve()
    for folder in (CACHE,RESULT):
        folder=folder.resolve()
        if folder==source.parent or source.parent in folder.parents: raise ValueError("Cache must be outside the dataset")
        folder.mkdir(exist_ok=True,parents=True)
    ids,t,y,raw=project(source)
    rows=[]; seen=0
    for pid in np.unique(ids):
        idx=np.flatnonzero(ids==pid); t0=t[idx]; idx=idx[np.argsort(t0,kind="stable")]
        times=t[idx]; x=raw[idx]; target=y[idx]
        duplicated=pd.Index(times).duplicated(keep=False)
        valid=np.isfinite(x).all(1)&~duplicated
        flat=np.ptp(np.nan_to_num(x),axis=1)==0
        # Treat a flat full second as lost waveform. Other quality indicators travel with the signal.
        valid&=~flat
        np.savez(CACHE/f"p{pid:02}.npz",t=times,y=target,raw=x,valid=valid,source_row=idx)
        rows.append(dict(patient=int(pid),rows=len(idx),usable=int(valid.sum()),nonfinite=int((~np.isfinite(x).all(1)).sum()),
            flat_finite=int((flat&np.isfinite(x).all(1)).sum()),duplicate_seconds=int(duplicated.sum()),
            missing_seconds=int(np.maximum(0,np.diff(times)-1).sum()),source_reversals=int((np.diff(t0)<0).sum()),
            unknown_labels=int((target<0).sum()),**{s:int((target==i).sum()) for i,s in enumerate(LABELS)}))
        seen+=len(idx)
    pd.DataFrame(rows).to_csv(RESULT/"cohort.csv",index=False)
    with source.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    summary=dict(source_bytes=source.stat().st_size,source_sha256=sha,source_md5=ARCHIVE_MD5,patients=len(rows),rows=seen,
        totals=pd.DataFrame(rows).drop(columns="patient").sum().astype(int).to_dict())
    (RESULT/"source_audit.json").write_text(json.dumps(summary,indent=2)); print(summary)
