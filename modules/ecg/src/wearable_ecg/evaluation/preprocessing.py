"""Independent development-only signal fidelity evaluation and graph export."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wearable_ecg.labels.ptbxl import cohort
from wearable_ecg.preprocessing.pipeline import process,corrupt,landmarks,morphology,SETTINGS
from wearable_ecg.evaluation.report import save_figure
from wearable_ecg.runtime import save_json


def evaluate_preprocessing(cache,output,n_per_class=64):
    cache=Path(cache); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(cache/"manifest.csv")
    eligible=cohort(frame,"lvh_vs_normal").query("strat_fold in [7,8]")
    waves=np.load(cache/"minimal_waves.npy",mmap_mode="r")
    controller=joblib.load(cache/"preprocessing/controller.joblib")
    rng=np.random.default_rng(906)
    selected=pd.concat([g.sample(min(n_per_class,len(g)),random_state=906) for _,g in eligible.groupby("label")])
    rows=[]; example=None
    # A dimensionless z-scored waveform has no physical mV-error endpoint.
    # Keep it available in the cache for a classifier-level amplitude ablation.
    variants=["minimal","conventional_fixed","swt","constrained"]
    for record in selected.itertuples():
        clean=np.array(waves[record.row_index]); peaks=landmarks(clean); ref=morphology(clean,peaks)
        if len(peaks)<3: continue
        for strength in (0.,1.,3.,6.):
            noisy=corrupt(clean,rng,strength)
            for variant in variants:
                out,feat,q,setting=process(noisy,controller,variant)
                phi=morphology(out,peaks)
                row={"ecg_id":record.ecg_id,"patient_id":record.patient_id,"label":record.label,
                     "variant":variant,"noise_strength":strength,"setting":int(setting),
                     "mae_mv":float(np.mean(abs(out-clean))),"rmse_mv":float(np.sqrt(np.mean((out-clean)**2))),
                     "r_relative_error":float(abs(phi[0]-ref[0])/max(abs(ref[0]),.2)),
                     "s_relative_error":float(abs(phi[1]-ref[1])/max(abs(ref[1]),.2)),
                     "qrs_area_error_mv_s":float(abs(phi[2]-ref[2])),
                     "qrs_width_proxy_error_ms":float(abs(phi[3]-ref[3])*1000),
                     "st_proxy_error_uv":float(abs(phi[4]-ref[4])*1000),
                     "t_proxy_error_mv":float(abs(phi[5]-ref[5])),
                     "t_polarity_changed":bool(np.sign(phi[5])!=np.sign(ref[5])),
                     "quality":float(np.mean(q))}
                row["within_proxy_budget"]=max(row["r_relative_error"],row["s_relative_error"])<=.05 and row["qrs_width_proxy_error_ms"]<=10 and row["st_proxy_error_uv"]<=20
                rows.append(row)
            if example is None and record.label==1 and strength==3:
                example=(record.ecg_id,clean,noisy,process(noisy,controller,"constrained")[0],process(noisy,controller,"swt")[0])
    table=pd.DataFrame(rows); table.to_csv(output/"preprocessing_development_pairs.csv",index=False)
    keys=["mae_mv","rmse_mv","r_relative_error","s_relative_error","qrs_width_proxy_error_ms","st_proxy_error_uv","within_proxy_budget"]
    summary=table.assign(within_proxy_budget=table.within_proxy_budget.astype(float)).groupby(["variant","noise_strength"])[keys].agg(["median","mean",lambda x:x.quantile(.95)])
    summary.columns=[f"{a}_{'p95' if b=='<lambda_0>' else b}" for a,b in summary.columns]
    summary.to_csv(output/"preprocessing_summary.csv")
    save_json(output/"preprocessing_scope.json",{"evaluation_folds":[7,8],"fit_folds":[1,2,3,4,5,6],
        "reference_records":int(table.ecg_id.nunique()),"noise":"synthetic baseline, 50-Hz interference, white noise; held-out RNG seed",
        "reference":"unaltered PTB-XL recording; not a certified clean signal",
        "fiducials":"Automatic engineering proxies; manual/cardiologist review and clinical delineation are outstanding",
        "zscore_note":"Excluded from physical mV/morphology-error comparisons because its output is dimensionless. Its cached signals are reserved for a classifier-level amplitude-removal ablation.",
        "selection_policy":"Evaluation describes the frozen controller; no retuning from these results."})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for variant in variants:
        subset=table[table.variant==variant].groupby("noise_strength")
        for ax,key in zip(axes,["mae_mv","st_proxy_error_uv","within_proxy_budget"]):
            values=subset[key].median() if key!="within_proxy_budget" else subset[key].mean()
            ax.plot(values.index,values.values,"o-",label=variant)
    for ax,title in zip(axes,["Median waveform MAE (mV)","Median ST proxy error (µV)","Fraction within all proxy budgets"]):
        ax.set(xlabel="Synthetic noise strength",title=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    title="Preprocessing fidelity · independent development folds 7–8"
    if table.loc[table.variant.eq("constrained"),"setting"].eq(0).all():
        title+="\nConstrained selector and minimal correction coincide in this pilot"
    fig.suptitle(title); fig.tight_layout()
    save_figure(fig,output/"preprocessing_fidelity")
    if example:
        ecg_id,clean,noisy,constrained,swt=example; t=np.arange(len(clean))/360
        fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)
        for ax,(name,trace) in zip(axes,[("Corrupted",noisy),("Constrained candidate",constrained),("Fixed robust baseline + SWT",swt)]):
            ax.plot(t,clean,color="#222222",lw=.8,label="PTB-XL reference")
            ax.plot(t,trace,color="#cc5d38",alpha=.8,lw=.7,label=name)
            ax.set(ylabel="mV",xlim=(2,6)); ax.legend(loc="upper right",fontsize=8); ax.grid(alpha=.15)
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(f"Development ECG {ecg_id} · example chosen before test scoring")
        fig.tight_layout(); save_figure(fig,output/"preprocessing_example")
