"""Add transparent interpretation, model cards and observed preprocessing status.

Reads frozen results only. Does not select or retrain a model from test scores.
"""
from pathlib import Path
import argparse,json,shutil,hashlib
import numpy as np
import pandas as pd


def finalize(cache,root,report):
    cache,root,report=map(Path,(cache,root,report))
    if not (root/"EVALUATION_COMPLETE.json").exists(): raise RuntimeError("Evaluation must finish first")
    for name in ("cohort_counts.csv","manifest_audit.json","source_integrity.json"):
        shutil.copy2(cache/name,report/name)
    if (root/"export_validation/inference_input_parity.json").exists():
        shutil.copy2(root/"export_validation/inference_input_parity.json",report/"inference_input_parity.json")
    records=pd.read_csv(cache/"manifest.csv")
    decisions=np.load(cache/"constrained_settings.npy")
    decision_summary=[]
    for partition,g in records.groupby("partition"):
        vals,counts=np.unique(decisions[g.row_index],return_counts=True)
        for setting,n in zip(vals,counts):
            decision_summary.append({"partition":partition,"setting":int(setting),"records":int(n),"fraction":float(n/len(g))})
    pd.DataFrame(decision_summary).to_csv(root/"preprocessing/controller_decisions_by_partition.csv",index=False)
    shutil.copy2(root/"preprocessing/controller_decisions_by_partition.csv",report/"preprocessing/controller_decisions_by_partition.csv")
    identity_only=bool(np.all(decisions==0))
    method_note=("**Observed preprocessing outcome:** the fitted controller selected identity/minimal correction for every cached recording. Consequently, the neural architecture results are effectively a minimally processed, calibrated-Lead-II baseline comparison. Independent synthetic-pair diagnostics also found no improvement of this fitted controller over identity. This run does not establish the proposed denoising contribution. The conservative baseline and reported negative preprocessing result are retained without changing the frozen comparison."
                 if identity_only else "Controller-setting frequencies are saved separately by partition; assess the independent fidelity report before claiming improvement.")
    path=report/"RESULTS.md"; text=path.read_text(encoding="utf-8")
    text=text.replace("Per-record z-scoring is included only as an explicit amplitude-removal ablation in preprocessing diagnostics; its absolute mV error is not comparable to calibrated denoising without that qualification.",
                      "A separate per-record z-score cache supports a future classifier-level amplitude-removal ablation. Its dimensionless output is excluded from physical mV-error comparisons.")
    if method_note not in text:
        text=text.replace("## Preprocessing evidence and scope\n", "## Preprocessing evidence and scope\n\n"+method_note+"\n")
    # Architecture selection is based only on internal development performance.
    selection=[]
    config=json.loads((root/"experiment.json").read_text())
    for task in config["tasks"]:
        candidates=[]
        for arch in config["architectures"]:
            chosen=[]
            for run in sorted((root/task).glob(f"{arch}__constrained__seed*")):
                info=json.loads((run/"COMPLETE.json").read_text())
                epoch=info["development"]["best_epoch"]
                history=pd.read_csv(run/"development/history.csv")
                row=history.loc[history.epoch.eq(epoch)].iloc[0]
                chosen.append([row.dev_sensitivity_at_90spec,row.dev_auprc,row.dev_auroc])
            mean=np.mean(chosen,axis=0)
            candidates.append({"task":task,"architecture":arch,"dev_sensitivity_mean":float(mean[0]),"dev_auprc_mean":float(mean[1]),"dev_auroc_mean":float(mean[2])})
        ordered=sorted(candidates,key=lambda r:(r["dev_sensitivity_mean"],r["dev_auprc_mean"]),reverse=True)
        for item in ordered:
            item["development_selected"]=item["architecture"]==ordered[0]["architecture"]
            selection.append(item)
    pd.DataFrame(selection).to_csv(report/"development_selection.csv",index=False)
    text=text.split("\n## Development-based candidate selection")[0]
    text+="\n## Development-based candidate selection\n\nThe following recommendations use mean internal-development sensitivity at the prespecified specificity, with average precision as tie-breaker. They were calculated from development histories, not from test rankings. All architectures and seeds remain reported. This is a research candidate choice; Nano timing and hardware validation may require a different backbone.\n\n"
    for item in selection:
        if item["development_selected"]:
            text+=f"- {item['task']}: **{item['architecture']}**, internal-development sensitivity {item['dev_sensitivity_mean']:.4f}, average precision {item['dev_auprc_mean']:.4f}.\n"
    text+="\n## Data and method attribution\n\nDataset: [PTB-XL 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), Wagner et al.; source data retain their supplied license. State-space background: [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893). The implemented real-exponential diagonal layer is an independent compact comparator, not a reproduction of the full S4D or Mamba systems.\n"
    path.write_text(text,encoding="utf-8")
    for task in config["tasks"]:
        # Aggregate statistics only; no record-level identifiers in shareable outputs.
        taskdir=report/task; taskdir.mkdir(parents=True,exist_ok=True)
        for name in ("ensemble_metrics.csv","paired_ensemble_bootstrap.json"):
            shutil.copy2(root/task/name,taskdir/name)
        ci_rows=[]; subgroup_rows=[]; runtime_rows=[]; confidence_rows=[]
        for arch in config["architectures"]:
            for run in sorted((root/task).glob(f"{arch}__constrained__seed*")):
                seed=json.loads((run/"config.json").read_text())["seed"]
                cis=json.loads((run/"test/patient_bootstrap_ci.json").read_text())
                for metric,limits in cis.items(): ci_rows.append({"architecture":arch,"seed":seed,"metric":metric,**limits})
                sub=pd.read_csv(run/"test/subgroups.csv"); sub["architecture"]=arch; sub["seed"]=seed; subgroup_rows.append(sub)
                latency=json.loads((run/"test/desktop_latency.json").read_text())
                run_metrics=json.loads((run/"test/metrics.json").read_text())
                predictions=pd.read_csv(run/"test/predictions.csv")
                for confidence,part in predictions[predictions.label.eq(1)].groupby("lvh_confidence"):
                    confidence_rows.append({"architecture":arch,"seed":seed,"annotation_likelihood":confidence,
                                            "positive_records":len(part),"positive_patients":part.patient_id.nunique(),
                                            "sensitivity":float(np.mean(part.probability>=run_metrics["threshold"]))})
                completed=json.loads((run/"COMPLETE.json").read_text())
                completed["test_evaluated"]=True
                completed["test_metrics_sha256"]=hashlib.sha256((run/"test/metrics.json").read_bytes()).hexdigest()
                (run/"COMPLETE.json").write_text(json.dumps(completed,indent=2)+"\n",encoding="utf-8")
                runtime_rows.append({"architecture":arch,"seed":seed,"parameters":run_metrics["parameters"],
                                     "selected_epochs":run_metrics["selected_epochs"],"training_seconds":run_metrics["training_seconds"],
                                     "desktop_model_p50_ms":latency["p50_ms"],"desktop_model_p95_ms":latency["p95_ms"]})
            bundle=root.parent.parent/"exports"/root.name/task/arch
            run=root/task/f"{arch}__constrained__seed42"
            met=json.loads((run/"test/metrics.json").read_text()); ci=json.loads((run/"test/patient_bootstrap_ci.json").read_text())
            contract=json.loads((bundle/"inference_contract.json").read_text())
            card=[f"# {arch}: {task}","","Research-only single-lead ECG annotation predictor. Prespecified seed 42; not selected by test score.","",
                  f"Input: Lead II, calibrated mV, 10 seconds, 360 Hz (3,600 samples). Parameters: {met['parameters']:,}.","",
                  f"Official fold-10 results: AUROC {met['auroc']:.4f} (patient-bootstrap 95% CI {ci['auroc']['low']:.4f}–{ci['auroc']['high']:.4f}); average precision {met['auprc']:.4f}; sensitivity {met['sensitivity']:.4f}; specificity {met['specificity']:.4f}. Threshold {met['threshold']:.6f} was fixed on fold 9.","",
                  "Probability calibration is cohort-specific. LVH-vs-normal case-control prevalence does not represent a wearable population. Neither a low score nor a high score establishes anatomical LVH status.","",method_note,"",
                  "Use `ECGPredictor` for matching preprocessing/calibration. ONNX accepts the preprocessed waveform and 16 matching voltage/morphology/quality features; do not feed raw ADC counts. Load only trusted local bundles.","",
                  "No clinical, BioAmp-domain, physical Jetson Nano or TensorRT FP16 validation has been completed. No cross-lead distillation or external foundation checkpoint was used.","",
                  f"ONNX SHA-256: `{contract['onnx_sha256']}`.",""]
            (bundle/"MODEL_CARD.md").write_text("\n".join(card),encoding="utf-8")
        pd.DataFrame(ci_rows).to_csv(taskdir/"patient_bootstrap_intervals.csv",index=False)
        runtimes=pd.DataFrame(runtime_rows)
        runtimes.to_csv(taskdir/"runtime_per_seed.csv",index=False)
        pd.DataFrame(confidence_rows).to_csv(taskdir/"annotation_confidence_sensitivity.csv",index=False)
        groups=pd.concat(subgroup_rows,ignore_index=True)
        groups.to_csv(taskdir/"subgroups_per_seed.csv",index=False)
        # The intended user population is older adults, so expose this explicitly.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        names=config["architectures"]
        for ax,metric in zip(axes,["auroc","sensitivity","specificity"]):
            for offset,subgroup,label,color in [(-.18,"age_under_65","Under 65","#4b91ad"),(.18,"age_65_plus","65+ (including encoded older ages)","#c2623e")]:
                selected=groups[groups.subgroup.eq(subgroup)].groupby("architecture")[metric].agg(["mean","std"]).reindex(names)
                ax.bar(np.arange(len(names))+offset,selected["mean"],.34,yerr=selected["std"],label=label,color=color,capsize=4)
            ax.set(xticks=range(len(names)),xticklabels=[n.replace("cnn_","CNN + ").replace("cnn","CNN") for n in names],ylim=(0,1),title=metric.upper())
            ax.tick_params(axis="x",labelrotation=12); ax.grid(axis="y",alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{task.replace('_',' ')} · age subgroups · mean ± seed SD · frozen fold 10")
        fig.tight_layout(); plotdir=report/"plots"/task; plotdir.mkdir(parents=True,exist_ok=True)
        fig.savefig(plotdir/"age_subgroups.png",dpi=180,bbox_inches="tight")
        fig.savefig(plotdir/"age_subgroups.pdf",bbox_inches="tight"); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,metric,title in zip(axes,["training_seconds","desktop_model_p95_ms"],
                                   ["Development + refit wall time (seconds)","Desktop model-only p95 inference (ms)"]):
            selected=runtimes.groupby("architecture")[metric].agg(["mean","std"]).reindex(names)
            ax.bar(range(len(names)),selected["mean"],yerr=selected["std"],color=["#1976a3","#c44c30","#618b32"],capsize=4)
            ax.set(xticks=range(len(names)),xticklabels=[n.replace("cnn_","CNN + ").replace("cnn","CNN") for n in names],title=title)
            ax.tick_params(axis="x",labelrotation=10); ax.grid(axis="y",alpha=.2)
        fig.suptitle("RTX 3050 laptop · mean ± seed SD · excludes acquisition/preprocessing; not Nano timing")
        fig.tight_layout(); fig.savefig(plotdir/"runtime_comparison.png",dpi=180,bbox_inches="tight")
        fig.savefig(plotdir/"runtime_comparison.pdf",bbox_inches="tight"); plt.close(fig)
    (root/"INTERPRETATION_COMPLETE.json").write_text(json.dumps({"controller_identity_only":identity_only,"report":str(path)},indent=2)+"\n")
    print("Interpretation and model cards saved:",path)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True); p.add_argument("--output",required=True); p.add_argument("--report-dir",required=True)
    a=p.parse_args(); finalize(a.cache,a.output,a.report_dir)
