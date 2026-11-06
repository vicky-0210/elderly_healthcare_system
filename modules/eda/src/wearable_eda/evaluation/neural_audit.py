"""Verify stored outer probabilities and measure host inference from prepared windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from wearable_eda.inference.neural import NeuralPredictor
from wearable_eda.training.neural_experiments import arguments
from wearable_eda.training.neural_suite import load_windows


def verify_checkpoint(folder, data, subject, device, batch_size):
    sd = folder / f"S{subject:02d}"
    select = np.flatnonzero(data["groups"] == subject)
    saved = pd.read_csv(sd/"predictions.csv")
    for column, source in (("start", "start"), ("end", "end"), ("y_true", "y")):
        if not np.array_equal(saved[column], data[source][select]):
            raise RuntimeError(f"Window identity mismatch in {sd} ({column})")
    inputs = (data["X"][select], data["F"][select], data["channels"], data["features"])
    predictor = NeuralPredictor(sd/"model.pt", device=device)
    probability = predictor.predict_windows(*inputs, batch_size=batch_size)
    expected = saved[[f"p{i}" for i in range(len(predictor.classes))]].to_numpy()
    error = float(np.abs(expected-probability).max())
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(f"Checkpoint replay failed in {sd}: {error}")
    cpu = probability if device=="cpu" else NeuralPredictor(sd/"model.pt").predict_windows(*inputs,batch_size=batch_size)
    disagreements = np.count_nonzero(cpu.argmax(1)!=expected.argmax(1))
    return dict(experiment=folder.name,subject=subject,windows=len(select),replay_device=device,
                max_error=error,cpu_max_error=float(np.abs(expected-cpu).max()),
                cpu_decision_differences=int(disagreements))


def benchmark(run, cache, device_threads=1):
    """The predeclared smallest subject is representative of graph shape and storage."""
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(device_threads)
    folders = sorted((run/"neural").glob("*/folds.csv"))
    measurements = []
    verifications = []
    active_key = None
    histories = []
    for foldfile in sorted(folders,key=lambda p:(json.dumps(arguments(json.loads((p.parent/"meta.json").read_text())),sort_keys=True),str(p))):
        folder = foldfile.parent
        spec = json.loads((folder/"meta.json").read_text())
        options = arguments(spec)
        key = json.dumps(options,sort_keys=True)
        if key != active_key:
            data = load_windows(cache,options)
            active_key = key
        folds = pd.read_csv(foldfile)
        subject = int(folds.subject.min())
        sd = folder/f"S{subject:02d}"
        predictor = NeuralPredictor(sd/"model.pt")
        sel = np.flatnonzero(data["groups"]==subject)
        results = [verify_checkpoint(folder,data,int(s),spec["device"],spec["batch_size"]) for s in folds.subject]
        maximum_error = max(item["max_error"] for item in results)
        verifications.extend(results)
        x, f = data["X"][sel[:1]],data["F"][sel[:1]]
        calls = 50
        with threadpool_limits(limits=device_threads):
            for _ in range(10):
                predictor.predict_windows(x,f,data["channels"],data["features"])
            times = []
            for _ in range(calls):
                t0 = time.perf_counter_ns()
                predictor.predict_windows(x,f,data["channels"],data["features"])
                times.append((time.perf_counter_ns()-t0)/1e6)
        p50,p95 = np.percentile(times,[50,95])
        params = sum(p.numel()*p.element_size() for p in predictor.model.parameters())
        buffers = sum(p.numel()*p.element_size() for p in predictor.model.buffers())
        row = dict(experiment=folder.name,subject=subject,batch=1,warmups=10,repeats=calls,
                   cpu_threads=device_threads,cpu_latency_p50_ms=p50,cpu_latency_p95_ms=p95,
                   parameter_bytes=params,buffer_bytes=buffers,prepared_input_bytes=x.nbytes+f.nbytes,
                   artifact_bytes=(sd/"model.pt").stat().st_size,replayed_folds=len(folds),
                   replay_max_abs_probability_error=maximum_error,
                   cpu=platform.processor(),scope="prepared windows including frozen scaler and temperature; excludes acquisition, feature extraction and disk IO")
        measurements.append(row)
        (folder/"benchmark.json").write_text(json.dumps(row,indent=2))
        for metric in ("cpu_latency_p50_ms","cpu_latency_p95_ms"):
            # Repeated table metadata, one benchmark for this architecture, not fold timings.
            folds[metric] = row[metric]
        folds["latency_benchmark_subject"] = subject
        folds.to_csv(folder/"folds.audited",index=False)
        (folder/"folds.audited").replace(foldfile)
        for s in folds.subject:
            fpath = folder/f"S{int(s):02d}"/"history.csv"
            history = pd.read_csv(fpath)
            history["subject"], history["experiment"] = int(s),folder.name
            histories.append(history)
    if not measurements:
        raise ValueError("No completed neural experiments")
    results = run/"neural_audit"
    results.mkdir(exist_ok=True)
    pd.DataFrame(verifications).to_csv(results/"checkpoint_replay.csv",index=False)
    pd.DataFrame(measurements).to_csv(results/"host_inference.csv",index=False)
    history = pd.concat(histories,ignore_index=True)
    history.to_csv(results/"learning_histories.csv.gz",index=False)
    plot_history(history,results)


def plot_history(history,results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"cnn_zero":"CNN","transformer_zero":"Transformer","multitcn_zero":"Multi-scale TCN",
             "pg_zero":"PG-MSTN","physics_zero":"PG + learned refiner","masked_multitcn_zero":"Masked-pretrained TCN"}
    h = history[(history.stage=="inner") & (history.phase=="supervised")]
    fig,axes = plt.subplots(1,2,figsize=(10.5,4.5),layout="constrained")
    for key,title in names.items():
        summary = h[h.experiment==key].groupby("epoch")[["loss","val_macro_f1"]].median()
        if summary.empty:
            continue
        for ax,metric in zip(axes,("loss","val_macro_f1")):
            ax.plot(summary.index,summary[metric],marker="o",ms=3,label=title)
    axes[0].set(ylabel="Training objective",xlabel="Epoch")
    axes[1].set(ylabel="Two-subject validation macro F1",xlabel="Epoch",ylim=(0,1))
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("Zero calibration · median across active outer folds at each epoch")
    fig.savefig(results/"learning_curves.png",dpi=160)
    plt.close(fig)


def physics_audit(run,cache):
    """Continuous held-out subject curves, grouped refit model; no scoring-label access."""
    from wearable_eda.datasets.wesad import subject_views
    from scipy.signal import find_peaks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((cache/"manifest.json").read_text())["subjects"]
    rows = []
    for exp in ("physics_zero","physics_no_aux_zero","physics_rest","pg_cv_physics_zero"):
        folder = run/"neural"/exp
        if not (folder/"folds.csv").exists():
            continue
        spec = json.loads((folder/"meta.json").read_text())
        for subject in manifest:
            s = subject["subject"]
            predictor = NeuralPredictor(folder/f"S{s:02d}"/"model.pt")
            mat, channels = subject_views(cache,subject,modalities=tuple(spec["modalities"]),
                calibration=spec["calibration"],decomposition=spec["decomposition"],filtering="conservative",quality="auto")
            x = mat[:,[channels.index(n) for n in predictor.channels]]
            scaler = predictor.scaler
            normalized = np.clip((x-scaler["xc"])/scaler["xs"],-30,30)
            with torch.inference_mode():
                scl,scr,resid = [c.squeeze(0).numpy() for c in predictor.model.refiner.components(torch.tensor(normalized.T[None]))]
            clean = x[:,predictor.channels.index("clean")]
            for kind,tonic,phasic,residual in (("input",mat[:,channels.index("tonic")],mat[:,channels.index("phasic")],np.zeros_like(clean)),
                                               ("learned",scl,scr,resid)):
                error = clean-tonic-phasic
                out = dict(experiment=exp,subject=s,component=kind,
                         component_rmse_uS=float(np.sqrt(np.mean(error**2))),
                         total_rmse_uS=float(np.sqrt(np.mean((error-residual)**2))),
                         residual_rms_uS=float(np.sqrt(np.mean(residual**2))),
                         tonic_mean_abs_second_difference_uS=float(np.abs(np.diff(tonic,n=2)).mean()),
                         phasic_peaks_per_minute=len(find_peaks(phasic,prominence=.03,distance=4)[0])*240/len(phasic),
                         nonnegative_phasic_fraction=float((phasic>=0).mean()))
                rows.append(out)
            if exp=="physics_zero" and s==2:
                # Fixed recording excerpt picked by index; no labels used to select it.
                seg = slice(len(clean)//2,len(clean)//2+960)
                t = np.arange(len(clean[seg]))/4
                fig,ax = plt.subplots(2,1,sharex=True,figsize=(9,5),layout="constrained")
                ax[0].plot(t,clean[seg],label="Clean EDA",color="#53778d")
                ax[0].plot(t,scl[seg],label="Learned tonic",color="#bf6935")
                ax[1].plot(t,scr[seg],label="Learned phasic",color="#267e69")
                ax[1].plot(t,resid[seg],label="Learned residual",color="#995152")
                for a in ax:
                    a.set_ylabel("Conductance (µS)")
                    a.legend()
                ax[1].set_xlabel("Time since recording midpoint (s)")
                fig.suptitle("Held-out subject S2 · zero-calibration constrained refiner")
                fig.savefig(run/"neural_audit"/"held_out_decomposition.png",dpi=150)
                plt.close(fig)
    if rows:
        pd.DataFrame(rows).to_csv(run/"neural_audit"/"continuous_decomposition.csv",index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("artifacts/eda/runs/wesad_loso_v1"))
    parser.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    args = parser.parse_args()
    benchmark(args.run,args.cache)
    physics_audit(args.run,args.cache)
