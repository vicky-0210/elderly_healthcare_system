"""Verify the full-data export on a cached, native-rate synchronized prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import tracemalloc

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODULE/"src"))
from wearable_eda.datasets.wesad import build_windows
from wearable_eda.inference.predictor import EDAPredictor, RATES, UNITS


def replay(export,cache,subject,report=None):
    model = EDAPredictor().load(export)
    cfg = model.contract
    if cfg["family"]=="neural":
        import torch
        torch.set_num_threads(1)
    data = build_windows(cache,modalities=tuple(cfg["modalities"]),calibration=cfg["calibration"])
    idx = np.flatnonzero(data["groups"]==subject)[0]
    end = data["end"][idx]
    streams = {}
    with np.load(cache/f"S{subject}.npz") as z:
        reserved = z["rest"].astype(float)/4
        for k in cfg["modalities"]:
            hz = RATES[k]
            x = z[k][:int(end*hz)]
            streams[k] = dict(unit=UNITS[k],samples=x/64 if k=="acc" else x,
                              timestamps=np.arange(len(x))/hz)
    window = dict(stream_id=f"replay-S{subject}",clock_domain="WESAD synchronized seconds",streams=streams)
    if cfg["calibration"] == "rest":
        if model.predict(window)["status"] != "insufficient_signal":
            raise AssertionError("Calibrated inference requires an explicitly designated rest interval")
        window["rest_interval_s"] = reserved.tolist()
    output = model.predict(window)
    if output["status"]!="ok":
        raise ValueError(f"Unexpected replay result {output}")
    if cfg["family"] == "neural":
        ref = model.model.predict_windows(data["X"][idx:idx+1],data["F"][idx:idx+1],data["channels"],data["features"])[0]
    else:
        from wearable_eda.training.classical import load_bundle, predict_windows
        reference = model.model
        if cfg.get("payload")=="linear.json":
            path = export/"model.joblib"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg["reference_sha256"]:
                raise ValueError("Research reference model checksum mismatch")
            reference = load_bundle(path)
        ref = predict_windows(reference,data["F"][idx:idx+1],data["features"])[0]
    diff = float(np.abs(ref-np.array(list(output["scores"].values()))).max())
    if diff>1e-5:
        raise ValueError(f"Replay diverges from offline processing by {diff}")
    durations = []
    for _ in range(25):
        begin = perf_counter()
        model.predict(window)
        durations.append(1000*(perf_counter()-begin))
    tracemalloc.start()
    model.predict(window)
    _,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    broken = {k:dict(v) for k,v in streams.items()}
    broken["eda"]["samples"] = np.full_like(streams["eda"]["samples"],np.nan)
    abstain = model.predict(dict(window,streams=broken))
    if abstain["status"]!="insufficient_signal" or abstain["scores"] is not None:
        raise AssertionError("Unusable EDA must produce an abstention")
    result = dict(prediction=output, missing_signal_check=abstain["status"],offline_max_abs_diff=diff,
                  prefix_seconds=float(end),timed_calls=len(durations),cpu_threads=1,cpu=platform.processor(),
                  prefix_to_prediction_p50_ms=float(np.median(durations)),prefix_to_prediction_p95_ms=float(np.quantile(durations,.95)),
                  python_traced_peak_bytes=peak,scope="Host CPU, complete prefix preprocessing and scoring after model load. Python-traced allocations exclude native framework workspaces.")
    if report:
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export",type=Path,default=Path("artifacts/eda/exports/wesad_loso_v1"))
    p.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    p.add_argument("--subject",type=int,default=2)
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        replay(args.export,args.cache,args.subject,args.report)
