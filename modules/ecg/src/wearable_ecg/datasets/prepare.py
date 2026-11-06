"""Read-only PTB-XL ingestion, verified source hashes and separate derived data."""
from __future__ import annotations
import json
import ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
import wfdb
from wearable_ecg.labels.ptbxl import annotate, cohort, assert_patient_folds
from wearable_ecg.preprocessing.pipeline import resample_mv, process, fit_controller, quality_features
from wearable_ecg.runtime import digest, save_json, separate_output


def load_cache(cache, variant="constrained"):
    cache=Path(cache)
    if not (cache/"COMPLETE.json").exists(): raise RuntimeError("Preparation did not finish")
    frame=pd.read_csv(cache/"manifest.csv")
    return frame, np.load(cache/f"{variant}_waves.npy",mmap_mode="r"), np.load(cache/f"{variant}_features.npy",mmap_mode="r")


def prepare(source, cache, workers=4):
    source=Path(source).resolve(); cache=separate_output(source,cache)
    if (cache/"COMPLETE.json").exists():
        prior=json.loads((cache/"COMPLETE.json").read_text())
        if prior["metadata_sha256"]!=digest(source/"ptbxl_database.csv"):
            raise ValueError("Source metadata changed; use a new cache directory")
        print("Reusing completed derived cache",cache,flush=True)
        return
    cache.mkdir(parents=True,exist_ok=True)
    expected={}
    for line in (source/"SHA256SUMS.txt").read_text().splitlines():
        sha, name=line.split(maxsplit=1); expected[name.lstrip("*")]=sha
    paths=["ptbxl_database.csv","scp_statements.csv","RECORDS"]
    inventory=[]
    def check(name):
        path=(source/name).resolve()
        if source not in path.parents: raise ValueError("Source path escapes dataset")
        h=digest(path)
        if name not in expected or h!=expected[name]: raise ValueError(f"Source checksum mismatch: {name}")
        st=path.stat()
        return {"path":name,"sha256":h,"size":st.st_size,"mtime_ns":st.st_mtime_ns}
    inventory.extend(check(p) for p in paths)
    frame=annotate(pd.read_csv(source/"ptbxl_database.csv"))
    statements=pd.read_csv(source/"scp_statements.csv",index_col=0)
    diagnostic_codes=set(statements.index[statements.diagnostic.eq(1)])
    frame["isolated_lvh"]=frame.scp_codes.map(lambda text: set(ast.literal_eval(text)) & diagnostic_codes == {"LVH"})
    assert_patient_folds(frame)
    n=len(frame)
    raw=np.lib.format.open_memmap(cache/"lead_ii_500hz_mv.npy",mode="w+",dtype="float32",shape=(n,5000))
    waves=np.lib.format.open_memmap(cache/"minimal_waves.npy",mode="w+",dtype="float32",shape=(n,3600))
    teacher=np.lib.format.open_memmap(cache/"teacher_12lead_360hz_mv.npy",mode="w+",dtype="float16",shape=(n,12,3600))
    rows=list(frame.itertuples(index=False))
    def read(row):
        base=row.filename_hr
        checks=[check(base+ext) for ext in (".hea",".dat")]
        x, info=wfdb.rdsamp(str(source/base))
        if info["fs"]!=500 or info["sig_len"]!=5000 or x.shape!=(5000,12):
            raise ValueError(f"Unexpected waveform dimensions: {base}")
        if any(unit!="mV" for unit in info["units"]): raise ValueError(f"Non-mV waveform: {base}")
        lead=info["sig_name"].index("II")
        # No imputation and no repair of corrupt files: fail with an audit trail.
        if not np.isfinite(x).all(): raise ValueError(f"Nonfinite samples: {base}")
        fixed_order=["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
        order=[info["sig_name"].index(k) for k in fixed_order]
        high=x[:,lead].astype(np.float32)
        canonical=resample_mv(x[:,order])
        usable=bool(np.std(high)>1e-5)
        return row.row_index, high, canonical, usable, checks
    usable=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,high,canonical,ok,checks in pool.map(read,rows):
            raw[i]=high; waves[i]=canonical[:,1]; teacher[i]=canonical.T
            usable.append(ok); inventory.extend(checks)
            if (i+1)%2000==0: print(f"Read and verified {i+1}/{n} records",flush=True)
    raw.flush(); waves.flush(); teacher.flush()
    frame["usable"]=usable
    frame.to_csv(cache/"manifest.csv",index=False)
    pd.DataFrame(inventory).to_csv(cache/"source_checksums_before.csv",index=False)
    counts=[]
    for task in ("lvh_vs_normal","lvh_vs_rest"):
        selected=cohort(frame,task)
        selected[["ecg_id","patient_id","strat_fold","partition","label","row_index"]].to_csv(cache/f"{task}_manifest.csv",index=False)
        for fold,g in selected.groupby("strat_fold"):
            counts.append({"task":task,"fold":int(fold),"records":len(g),"patients":g.patient_id.nunique(),"positive":int(g.label.sum()),"negative":int((1-g.label).sum())})
    pd.DataFrame(counts).to_csv(cache/"cohort_counts.csv",index=False)
    inner=cohort(frame,"lvh_vs_normal").query("strat_fold <= 6")
    controller=fit_controller(np.asarray(waves[inner.row_index]),inner.label.to_numpy(),cache/"preprocessing")
    del raw, teacher
    for variant in ("minimal","conventional_fixed","conventional_zscore","swt","constrained"):
        dst=waves if variant=="minimal" else np.lib.format.open_memmap(cache/f"{variant}_waves.npy",mode="w+",dtype="float32",shape=(n,3600))
        feats=np.lib.format.open_memmap(cache/f"{variant}_features.npy",mode="w+",dtype="float32",shape=(n,16))
        choices=np.zeros(n,dtype=np.int8)
        def transform(i): return i, process(np.asarray(waves[i]),controller,variant)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,(y,v,q,c) in pool.map(transform,range(n)):
                dst[i]=y; feats[i]=v; choices[i]=c
                if (i+1)%5000==0: print(f"Preprocessed {variant}: {i+1}/{n}",flush=True)
        dst.flush(); feats.flush()
        np.save(cache/f"{variant}_settings.npy",choices)
        if variant!="minimal": del dst
    save_json(cache/"source_integrity.json",{"files_verified":len(inventory),"official_checksums":"SHA256SUMS.txt",
        "read_only_source":str(source),"source_hash_manifest": "source_checksums_before.csv",
        "metadata_sha256":digest(source/"ptbxl_database.csv"),"excluded_flat_records":int((~frame.usable).sum()),
        "contradictory_ecg_ids":frame.loc[frame.contradiction,"ecg_id"].tolist(),"patient_fold_overlap":0})
    save_json(cache/"COMPLETE.json",{"metadata_sha256":digest(source/"ptbxl_database.csv"),"records":n,
        "source":str(source),"source_rate_hz":500,"model_rate_hz":360,"lead":"II","duration_seconds":10,
        "controller_fit_folds":[1,2,3,4,5,6],"preparation_version":1})
    print("Preparation complete",cache,flush=True)


def verify_unchanged(source, cache, output):
    """Re-hash all files used, including waveform payloads, after experiments."""
    source=Path(source); before=pd.read_csv(Path(cache)/"source_checksums_before.csv")
    mismatches=[]
    for row in before.itertuples(index=False):
        if digest(source/row.path)!=row.sha256: mismatches.append(row.path)
    save_json(output,{"files_rehashed":len(before),"changed_files":mismatches,"source_unchanged":not mismatches})
    if mismatches: raise ValueError("Source integrity check failed")


def finalize_manifest_audit(source,cache):
    """Audit duplicate payloads and derive diagnostic-only subgroup flags."""
    source,cache=Path(source),Path(cache)
    frame=pd.read_csv(cache/"manifest.csv")
    statements=pd.read_csv(source/"scp_statements.csv",index_col=0)
    diagnostic=set(statements.index[statements.diagnostic.eq(1)])
    frame["isolated_lvh"]=frame.scp_codes.map(lambda s:(set(ast.literal_eval(s)) & diagnostic)=={"LVH"})
    frame.to_csv(cache/"manifest.csv",index=False)
    hashes=pd.read_csv(cache/"source_checksums_before.csv")
    hashes=hashes[hashes.path.str.endswith(".dat")].copy()
    hashes["filename_hr"]=hashes.path.str[:-4]
    linked=frame.merge(hashes,on="filename_hr",validate="one_to_one")
    duplicates=linked[linked.sha256.duplicated(keep=False)]
    crossing=duplicates.groupby("sha256").strat_fold.nunique()
    duplicates[["ecg_id","patient_id","strat_fold","sha256"]].to_csv(cache/"duplicate_waveform_payloads.csv",index=False)
    save_json(cache/"manifest_audit.json",{"source_records":len(frame),"patient_overlap":0,
        "duplicate_payload_records":len(duplicates),"duplicate_payload_groups_crossing_folds":int((crossing>1).sum()),
        "isolated_lvh_diagnostic_records":int((frame.isolated_lvh & ~frame.contradiction).sum()),
        "manifest_sha256":digest(cache/"manifest.csv")})
    if (crossing>1).any(): raise ValueError("Identical waveform payloads cross official folds; protocol review required")
