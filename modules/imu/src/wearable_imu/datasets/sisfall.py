"""Validate the local SisFall release and cache the six selected native channels."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

RATE = 200
COUNTS_TO_PHYSICAL = np.array([32/8192]*3 + [4000/65536]*3, dtype=np.float32)
PATTERN = re.compile(r"([DF]\d{2})_(S[AE]\d{2})_(R\d{2})\.txt", re.I)
CACHE = Path("data/imu/processed/sisfall_v1")
RESULTS = Path("research/results/sisfall_v1")


def read_counts(contents):
    if re.search(rb"[^0-9,+;\s-]",contents):
        raise ValueError("Expected signed integer counts")
    if re.search(rb";[^\r\n\t ]",contents):
        raise ValueError("Unexpected data after a sample terminator")
    raw = np.loadtxt(BytesIO(contents.strip().replace(b";",b" ")),delimiter=",",dtype=np.int32,ndmin=2)
    if raw.shape[1]!=9 or not len(raw) or contents.count(b";")!=len(raw):
        raise ValueError("Expected nine integer channels and one terminator per sample")
    if np.abs(raw.astype(np.int64)).max()>32768 or np.any(raw>32767):
        raise ValueError("Value outside int16 storage range")
    return raw


def make_folds(frame):
    rng=np.random.default_rng(1701)
    assignments={}
    # Balance fall-contributing and ADL-only participants across the outer groups.
    has_falls = frame.groupby("subject").y.max()
    for _, roster in has_falls.groupby(has_falls):
        ids=rng.permutation(sorted(roster.index))
        for i,s in enumerate(ids):
            assignments[s]=i%5
    return frame.subject.map(assignments)


def audit(source,cache,out):
    source,cache,out=map(lambda p:Path(p).resolve(),(source,cache,out))
    if not (source/"Readme.txt").is_file() and (source/"SisFall_dataset").is_dir():
        source=source/"SisFall_dataset"
    if not (source/"Readme.txt").is_file():
        raise ValueError("Expected the release directory containing Readme.txt and SA/SE folders")
    if any(source==p or source in p.parents or p in source.parents for p in (cache,out)):
        raise ValueError("Keep every derived file outside the source dataset tree")
    cache.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in source.glob("*/*.txt") if p.parent.name.startswith(("SA","SE")))
    if not files:
        raise ValueError("No subject trials")
    records,errors,offset,seen=[],[],0,{}
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        stream=Path(tmp)/"native.i16"
        with stream.open("wb") as output:
            for path in files:
                try:
                    match=PATTERN.fullmatch(path.name)
                    if not match or match[2]!=path.parent.name:
                        raise ValueError("Unexpected trial identity")
                    activity,subject,trial=match.groups()
                    if not 1<=int(activity[1:])<=(19 if activity[0]=="D" else 15):
                        raise ValueError("Unknown activity")
                    blob=path.read_bytes()
                    digest=sha256(blob).hexdigest()
                    raw=read_counts(blob)
                    six=np.ascontiguousarray(raw[:,:6],dtype="<i2")
                    fingerprint=sha256(six.tobytes()).hexdigest()
                    if fingerprint in seen:
                        raise ValueError("Exact duplicate six-axis trial of "+seen[fingerprint])
                    seen[fingerprint]=path.stem
                    n=len(six)
                    if n<RATE:
                        raise ValueError("Less than one second of data")
                    acc=six[:,:3].astype("f")*COUNTS_TO_PHYSICAL[:3]
                    gyro=six[:,3:].astype("f")*COUNTS_TO_PHYSICAL[3:]
                    duplicate=np.all(six[1:]==six[:-1],axis=1)
                    rails=((six[:,:3]<=-4096)|(six[:,:3]>=4095)).any(1)
                    gyro_rails=((six[:,3:]==-32768)|(six[:,3:]==32767)).any(1)
                    records.append(dict(id=path.stem,subject=subject,activity=activity,trial=trial,
                        y=int(activity.startswith("F")),older=subject.startswith("SE"),offset=offset,n=n,
                        seconds=n/RATE,source_bytes=len(blob),source_sha256=digest,signal_sha256=fingerprint,
                        repeat_rows=int(duplicate.sum()),acc_rail_samples=int(rails.sum()),
                        gyro_rail_samples=int(gyro_rails.sum()),peak_g=float(np.linalg.norm(acc,axis=1).max()),
                        peak_dps=float(np.linalg.norm(gyro,axis=1).max()),
                        max_step_g=float(np.linalg.norm(np.diff(acc,axis=0),axis=1).max())))
                    six.tofile(output)
                    offset+=n
                except (ValueError,OSError,UnicodeError) as exc:
                    errors.append(dict(file=str(path.relative_to(source)),reason=str(exc)))
        frame=pd.DataFrame(records)
        if not len(frame) or frame.id.duplicated().any():
            raise ValueError("No usable cohort or duplicated identities")
        frame["fold"]=make_folds(frame)
        # Save the model inputs and index together only after all trials have been checked.
        import shutil
        shutil.copyfile(stream,cache/"native.pending")
        (cache/"native.pending").replace(cache/"native.i16")
        frame.to_csv(cache/"manifest.csv",index=False)
    pd.DataFrame(errors,columns=["file","reason"]).to_csv(out/"excluded.csv",index=False)
    frame.groupby(["activity","y"]).agg(recordings=("id","size"),subjects=("subject","nunique"),
        min_s=("seconds","min"),median_s=("seconds","median"),max_s=("seconds","max"),hours=("seconds",lambda x:x.sum()/3600)).to_csv(out/"activity_audit.csv")
    frame.groupby(["subject","fold","older","y"]).agg(recordings=("id","size"),hours=("seconds",lambda x:x.sum()/3600)).to_csv(out/"subject_audit.csv")
    report=dict(source_files=len(files),included=len(frame),excluded=len(errors),participants=frame.subject.nunique(),
        fall_subjects=sorted(frame[frame.y==1].subject.unique()),older_fall_subjects=sorted(frame[(frame.y==1)&frame.older].subject.unique()),
        adl=int((frame.y==0).sum()),fall=int(frame.y.sum()),adl_hours=frame.loc[frame.y==0,"seconds"].sum()/3600,
        hours=frame.seconds.sum()/3600,seconds_range=[frame.seconds.min(),frame.seconds.max()],
        acc_rail_samples=int(frame.acc_rail_samples.sum()),gyro_rail_samples=int(frame.gyro_rail_samples.sum()),
        repeated_six_axis_rows=int(frame.repeat_rows.sum()),native_rows=offset,
        manifest_sha256=sha256((cache/"manifest.csv").read_bytes()).hexdigest())
    (out/"dataset_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def load_cache(cache=CACHE):
    cache=Path(cache)
    manifest=pd.read_csv(cache/"manifest.csv")
    counts=np.memmap(cache/"native.i16",dtype="<i2",mode="r").reshape(-1,6)
    if len(counts)!=manifest.n.sum():
        raise ValueError("Native cache and metadata do not match")
    return manifest,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--out",type=Path,default=RESULTS)
    args=parser.parse_args()
    audit(args.source,args.cache,args.out)
