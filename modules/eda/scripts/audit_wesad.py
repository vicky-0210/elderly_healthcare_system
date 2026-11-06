"""Audit released wrist data and compare fixed physiological decompositions."""
import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from threadpoolctl import threadpool_limits

from wearable_eda.datasets.wesad import build_windows
from wearable_eda.preprocessing.wrist import transform_recording, decompose

METHODS = ("mcsaed","cvxeda","choi","sparse")
CONDITIONS = ("Baseline","Stress","Amusement","Meditation")
COLORS = ("#40627d","#be5545","#b18b37","#51866d")


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8 << 20),b""):
            h.update(chunk)
    return h.hexdigest()


def measure(clean, tonic, phasic):
    residual = clean-tonic-phasic
    a, b = np.std(clean), np.std(phasic)
    result = dict(reconstruction_rmse_uS=np.sqrt(np.mean(residual**2)),
                  tonic_curvature_uS_per_s2=np.abs(np.diff(tonic,n=2)*16).mean(),
                  phasic_sd_uS=b, finite_fraction=np.isfinite(tonic+phasic).mean())
    # Matched observed deflections are a morphology proxy, since WESAD has no SCR ground truth.
    ref, r = signal.find_peaks(clean, prominence=.03,distance=4)
    for thresh in (.02,.03,.05):
        q, qp = signal.find_peaks(phasic, prominence=thresh,distance=4)
        match = []
        # One-to-one temporal matching, at most one second of peak displacement.
        i = 0
        for j in range(len(ref)):
            while i<len(q) and q[i]<ref[j]-4:
                i += 1
            if i<len(q) and abs(q[i]-ref[j])<=4:
                match.append(qp["prominences"][i]/r["prominences"][j])
                i += 1
        tag = f"{thresh:.2f}".replace(".","p")
        result[f"matched_raw_peak_fraction_{tag}"] = len(match)/max(1,len(ref))
        result[f"median_matched_amplitude_ratio_{tag}"] = float(np.median(match)) if match else np.nan
        result[f"scr_peaks_per_min_{tag}"] = len(q)*240/len(clean)
    return result


def plot_examples(output, streams, counts):
    vals,names = transform_recording(streams,modalities=("eda","acc","temp","bvp"))
    view = dict(zip(names,vals.T))
    fig,axes = plt.subplots(4,3,figsize=(13,8.5),layout="constrained",sharex=True)
    for code,(label,color) in enumerate(zip(CONDITIONS,COLORS),1):
        runs = np.flatnonzero(np.diff(np.r_[False,counts[:,code]==175,False])).reshape(-1,2)
        a,b = max(runs,key=lambda edge:edge[1]-edge[0])
        a = int((a+b)//2-180)
        idx=slice(a,a+360)
        t=np.arange(360)/4
        ax=axes[code-1]
        ax[0].plot(t,view["raw"][idx],color=".6",lw=1,label="Raw")
        ax[0].plot(t,view["tonic"][idx],color=color,lw=1.5,label="Tonic estimate")
        ax[1].plot(t,view["phasic"][idx],color=color,lw=1.3)
        ax[2].plot(t,view["quality"][idx],color="#283e4d",lw=1,label="EDA quality")
        ax[2].plot(t,view["motion"][idx],color="#bf642f",lw=1,label="Motion intensity")
        ax[0].set_ylabel(label+"\nµS")
        ax[2].set_ylim(-.04,1.06)
    for a,h in zip(axes[0],("Wrist EDA and estimated SCL (µS)","Estimated SCR (µS)","Motion and quality (0–1)")):
        a.set_title(h)
    for a in axes[-1]:
        a.set_xlabel("Seconds into excerpt")
    axes[0,0].legend(fontsize=8)
    axes[0,2].legend(fontsize=8)
    fig.suptitle("S2 wrist physiology during four protocol conditions\n90-second excerpts centered in each participant condition",fontsize=15)
    fig.savefig(output/"protocol_examples.png",dpi=150)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    p.add_argument("--output",type=Path,default=Path("artifacts/eda/reports/wesad_stress/preprocessing"))
    p.add_argument("--publish",type=Path,default=None)
    args=p.parse_args()
    for destination in (args.output,args.publish):
        if destination and (args.source.resolve() == destination.resolve() or args.source.resolve() in destination.resolve().parents):
            raise ValueError("Outputs must be separate from the source dataset")
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.cache/"manifest.json").read_text())["subjects"]
    from cvxopt import solvers
    official_qp=solvers.qp
    statuses=[]
    def logging_qp(*qp_args,**kwargs):
        result=official_qp(*qp_args,**kwargs)
        statuses.append({k:result.get(k) for k in ("status","iterations","gap","primal infeasibility","dual infeasibility")})
        return result
    audit=[]
    morphology=[]
    windows=build_windows(args.cache)
    ages=[]
    genders=[]
    with threadpool_limits(limits=1),patch.object(solvers,"qp",logging_qp):
        for row in rows:
            sid=row["subject"]
            raw=args.source/f"S{sid}"/f"S{sid}_readme.txt"
            metadata=raw.read_text(encoding="latin1")
            ages.append(int(re.search(r"(?im)^Age:\s*(\d+)",metadata).group(1)))
            genders.append(re.search(r"(?im)^Gender:\s*(\w+)",metadata).group(1).lower())
            verified=checksum(args.source/row["file"])==row["sha256"]
            if not verified:
                raise ValueError(f"Source hash mismatch S{sid}")
            with np.load(args.cache/f"S{sid}.npz") as z:
                streams={k:z[k] for k in ("eda","acc","temp","bvp")}
                counts=z["counts"]
            if sid==2:
                plot_examples(out,streams,counts)
            v,names=transform_recording(streams,modalities=("eda","acc"))
            solo,_=transform_recording(streams)
            get=lambda key:v[:,names.index(key)]
            q,artifact,motion=get("quality"),get("artifact"),get("motion")
            eda=streams["eda"]
            assigned=windows["y"][windows["groups"]==sid]
            rawstarts=[s["raw_e4_start_unix"] for s in row["streams"].values()]
            df=np.diff(eda,prepend=eda[0])
            a=dict(subject=sid,record_s=len(eda)/4,sha256_verified=verified,
                   raw_start_spread_s=max(rawstarts)-min(rawstarts),
                   label_tail_samples=row["label_samples"]-len(eda)*175,
                   eda_min_uS=float(np.nanmin(eda)),eda_max_uS=float(np.nanmax(eda)),
                   physical_range_exceedances=int(((eda<0)|(eda>100)).sum()),
                   mean_quality=float(q.mean()),motion_low_quality_fraction=float((q<.5).mean()),
                   intrinsic_low_quality_fraction=float((solo[:,names.index("quality")]<.5).mean()),
                   zero_motion_low_quality_fraction=float(np.mean(q[motion<.1]<.5)) if (motion<.1).any() else None,
                   active_motion_low_quality_fraction=float(np.mean(q[motion>=.5]<.5)) if (motion>=.5).any() else None,
                   motion_change_spearman=float(stats.spearmanr(abs(df),motion).statistic),
                   calibration_start_s=row["calibration_start_s"],calibration_end_s=row["calibration_end_s"],
                   binary_baseline_windows=int((assigned==0).sum()),binary_stress_windows=int((assigned==1).sum()))
            for k,st in row["streams"].items():
                a[k.lower()+"_nonfinite"]=st["nonfinite"]
                a[k.lower()+"_trimmed"]=st["trimmed_tail_samples"]
                a[k.lower()+"_hz"]=st["rate"]
            for lab,d in enumerate(row["label_duration_s"]):
                a[f"protocol_{lab}_s"]=d
            audit.append(a)
            clean=solo[:,names.index("clean")].astype(float)
            confidence=solo[:,names.index("artifact")].astype(float)
            for method in METHODS:
                start=time.perf_counter()
                t,ph=decompose(clean,confidence,method)
                cost=time.perf_counter()-start
                morphology.append(dict(subject=sid,method=method,seconds=cost,
                                       ms_per_minute=cost*240000/len(clean),**measure(clean,t,ph)))
            print(f"S{sid} verified and benchmarked",flush=True)
    frame=pd.DataFrame(audit)
    frame.to_csv(out/"recording_audit.csv",index=False)
    rawmet=pd.DataFrame(morphology)
    rawmet.to_csv(out/"decomposition_by_subject.csv",index=False)
    grouped=rawmet.groupby("method").mean(numeric_only=True).drop(columns="subject")
    grouped.to_csv(out/"decomposition_summary.csv")
    pd.DataFrame(statuses).to_csv(out/"cvxeda_solver_diagnostics.csv",index=False)
    bysubject=np.arange(len(frame))
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    bottom=np.zeros(len(frame))
    for i,(color,label) in enumerate(zip(COLORS,CONDITIONS),1):
        y=frame[f"protocol_{i}_s"]/60
        ax.bar(bysubject,y,bottom=bottom,label=label,color=color)
        bottom+=y
    ax.set(xticks=bysubject,xticklabels=["S"+str(x) for x in frame.subject],ylabel="Minutes",
           title="Protocol time by WESAD participant")
    ax.legend(ncols=4,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.08))
    fig.savefig(out/"cohort_conditions.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,col,title in zip(axes,("reconstruction_rmse_uS","matched_raw_peak_fraction_0p03","ms_per_minute"),
                            ("Residual RMSE (µS)","Fraction of raw peaks matched","Compute (ms per minute)")):
        for j,method in enumerate(METHODS):
            pixels=rawmet[rawmet.method==method][col].to_numpy()
            ax.scatter(np.full(len(pixels),j),pixels,s=12,alpha=.5,color=COLORS[j])
            ax.plot([j-.24,j+.24],[pixels.mean()]*2,lw=2,color="black")
        ax.set(xticks=np.arange(4),xticklabels=METHODS,title=title)
        ax.tick_params(axis="x",rotation=25)
        if col=="ms_per_minute":
            ax.set_yscale("log")
    fig.suptitle("Continuous EDA decomposition on 15 participants",fontsize=15)
    fig.savefig(out/"decomposition_comparison.png",dpi=150)
    plt.close(fig)
    summary=dict(participants=len(ages),age_min=min(ages),age_max=max(ages),age_mean=np.mean(ages),
                 age_sd=np.std(ages,ddof=1),gender_counts=dict(pd.Series(genders).value_counts().astype(int).items()),
                 recordings_hours=frame.record_s.sum()/3600,sha256_verified=int(frame.sha256_verified.sum()),
                 binary_baseline_windows=int(frame.binary_baseline_windows.sum()),binary_stress_windows=int(frame.binary_stress_windows.sum()),
                 excluded_context_codes=[0,3,4,5,6,7],binary_purity=.8,
                 cvxeda_qp_states=pd.Series([s["status"] for s in statuses]).value_counts().to_dict())
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2,default=lambda z:z.item())+"\n")
    if args.publish:
        args.publish.mkdir(parents=True,exist_ok=True)
        for name in ("recording_audit.csv","decomposition_summary.csv","decomposition_by_subject.csv",
                     "audit_summary.json","cohort_conditions.png","decomposition_comparison.png","protocol_examples.png"):
            shutil.copyfile(out/name,args.publish/name)
    print(summary)


if __name__=="__main__":
    main()
