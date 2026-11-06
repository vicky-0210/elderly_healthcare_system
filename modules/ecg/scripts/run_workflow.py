"""Complete read-only-source experiment, including results and export bundles."""
from pathlib import Path
import sys,json,argparse,subprocess,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"packages/common/src"))
from wearable_ecg.datasets.prepare import prepare,verify_unchanged,finalize_manifest_audit
from wearable_ecg.training.runner import train_suite
from wearable_ecg.evaluation.report import evaluate_suite
from wearable_ecg.evaluation.preprocessing import evaluate_preprocessing
from wearable_ecg.inference.export import export_bundle
from wearable_ecg.runtime import save_json,digest


def write_report(root,report_dir,exports):
    import pandas as pd
    import numpy as np
    root=Path(root); report_dir=Path(report_dir); report_dir.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(root/"comparison_summary.csv")
    allrows=pd.read_csv(root/"comparison_per_seed.csv")
    lines=["# PTB-XL Lead II LVH-pattern model comparison", "",
           "This report contains measured results from the local PTB-XL 1.0.3 experiment. The target is the LVH ECG diagnostic statement, not imaging-confirmed anatomical LVH.", "",
           "## Experimental design", "",
           "500-Hz physical waveforms were read without modification and resampled to 360 Hz in a separate cache. Models receive a complete 10-second Lead II window and same-lead derived features. Official patient folds were retained: 1–6 for development training, 7–8 for epoch selection, 1–8 for refitting, 9 for probability calibration and thresholds, and 10 for one frozen final comparison. Five random seeds (42–46) were run per architecture and task. Class-weighted BCE, AdamW, batch size 32, a 100-epoch ceiling and patience 15 were shared. No demographics, text reports or diagnostic codes were model inputs.", "",
           "CNN, CNN–Transformer and CNN–diagonal-SSM share a multiscale stem, separate physical-voltage branch and train-scaled same-lead morphology features. The SSM is a real diagonal time-invariant exponential-kernel model; it is not the input-selective Mamba architecture.", "",
           "## Individual-model results", "",
           "Values are mean ± sample standard deviation across five separately trained seeds. Thresholds are selected for at least 90% specificity on fold 9; test specificity is allowed to differ. Average precision is reported as AUPRC. Ensembles are separately labeled in their files and plots.", "",
           "| Task | Architecture | AUROC | AUPRC | Sensitivity | Specificity | F1 | Brier | Parameters |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _,r in summary.iterrows():
        value=lambda k:f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}"
        lines.append(f"| {r.task} | {r.architecture} | {value('auroc')} | {value('auprc')} | {value('sensitivity')} | {value('specificity')} | {value('f1')} | {value('brier')} | {int(r.parameters_mean):,} |")
    lines.extend(["", "## Controls", "", "Fixed-configuration voltage/morphology logistic regression and random forest, age/sex-only logistic regression, and a quality-only logistic control use the same cohort/folds. These are single deterministic control runs, not five-seed deep-learning estimates.", "",
                  "| Task | Control | AUROC | AUPRC | Sensitivity | Specificity |", "|---|---|---:|---:|---:|---:|"])
    for _,r in allrows[~allrows.architecture.isin(["cnn","cnn_transformer","cnn_ssm"])].iterrows():
        lines.append(f"| {r.task} | {r.architecture} | {r.auroc:.4f} | {r.auprc:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} |")
    lines.extend(["", "## Preprocessing evidence and scope", "",
        "The candidate uses a training-only decision tree to select identity, robust baseline correction, or bounded stationary-wavelet correction with conditional mains subtraction. Selection targets synthetic-pair reconstruction error subject to engineering amplitude, duration and ST proxy limits. Independent fold-7/8 fidelity results are saved, including failed budgets and clean-signal changes. Automatic landmarks are not a validated clinical delineator. The candidate is an initial implementation of the proposed hypothesis; neither novelty nor improved clinical morphology preservation has been established.", "",
        "The controller may choose identity frequently. That is an observed limitation, not evidence that artifact removal is unnecessary. A separate per-record z-score cache supports a future classifier-level amplitude-removal ablation. Its dimensionless output is excluded from physical mV-error comparisons.", "",
        "## Saved evidence", "",
        "- `comparison_per_seed.csv` and `comparison_summary.csv`: every neural run and aggregate results.",
        "- `plots/<task>/`: ROC/PR/reliability curves, confusion matrices, seed comparisons, learning curves, selective-risk curves and subgroup views, as PNG and PDF.",
        "- Per-run `test/`: record-level predictions, patient-cluster bootstrap 95% intervals (1,000 resamples), subgroup metrics and desktop latency.",
        "- Task directories: paired patient-bootstrap differences between five-seed ensembles. These intervals condition on trained models and do not replace training-seed variability.",
        "- Each run: development/refit histories, checkpoint, configuration, calibration parameters, frozen thresholds and model checksum.",
        "- `source_integrity_after.json`: post-experiment rehash of every source file used.",
        "- Export bundles: PyTorch, fixed-shape ONNX, identical preprocessing controller, calibration, input contract and measured CPU ONNX parity.", "",
        "## Interpretation limits and remaining protocol work", "",
        "This completes the supervised architecture comparison stage. Cross-lead teachers and reliability-gated distillation, JEPA/pretraining, a full classifier-level preprocessing/KD ablation matrix, a clinically reviewed delineator, age/sex-matched sensitivity analysis, real-noise stress tests, and external Georgia/BioAmp validation are not claimed as completed experiments. No external pretrained checkpoint is used. These additions must have a separate frozen protocol; the present test results must not guide retuning on fold 10.", "",
        "The original Nano, ADC calibration, sensor placement and real wearable-domain clinical/reference labels were not available for validation. Desktop latency excludes acquisition and preprocessing and does not establish Nano suitability. A 10-second buffered window incurs acquisition delay; it is not sample-causal inference. No anatomical LVH diagnosis or exclusion is supported.", "",
        "## Local run locations", "", f"Full local outputs: `{root}`.", ""])
    for out in exports: lines.append(f"- Export: `{out}`")
    (report_dir/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for name in ("comparison_per_seed.csv","comparison_summary.csv","environment.json","experiment.json","source_integrity_after.json"):
        shutil.copy2(root/name,report_dir/name)
    for src in (root/"plots").rglob("*"):
        if src.is_file():
            dst=report_dir/"plots"/src.relative_to(root/"plots"); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for src in (root/"preprocessing").glob("*"):
        if src.suffix in (".png",".pdf") or src.name in ("preprocessing_summary.csv","preprocessing_scope.json"):
            dst=report_dir/"preprocessing"/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    # Only aggregate evidence is copied to the versionable report, not records.


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path); p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path); p.add_argument("--config",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    a=p.parse_args(); cfg=json.loads(a.config.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    prepare(a.source,a.cache)
    finalize_manifest_audit(a.source,a.cache)
    snapshot=a.output/"source_snapshot"
    source_root=Path(__file__).resolve().parents[1]
    if not (a.output/"FROZEN_FOR_TEST.json").exists():
        shutil.copytree(source_root/"src",snapshot/"src",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(source_root/"scripts",snapshot/"scripts",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    lock=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (a.output/"requirements.lock.txt").write_text(lock,encoding="utf-8")
    if not (a.output/"preprocessing/preprocessing_scope.json").exists():
        evaluate_preprocessing(a.cache,a.output/"preprocessing")
    if not (a.output/"FROZEN_FOR_TEST.json").exists(): train_suite(a.cache,a.output,cfg)
    evaluate_suite(a.cache,a.output)
    verify_unchanged(a.source,a.cache,a.output/"source_integrity_after.json")
    exports=[]
    for task in cfg["tasks"]:
        for architecture in cfg["architectures"]:
            # Prespecified seed, not selected by best test score.
            run=a.output/task/f"{architecture}__constrained__seed42"
            destination=a.output.parent.parent/"exports"/a.output.name/task/architecture
            exports.append(export_bundle(run,a.cache,destination))
    write_report(a.output,a.report_dir,exports)
    subprocess.check_call([sys.executable,str(source_root/"scripts/finalize_evidence.py"),"--cache",str(a.cache),
                           "--output",str(a.output),"--report-dir",str(a.report_dir)])
    save_json(a.output/"WORKFLOW_COMPLETE.json",{"report":str(a.report_dir/"RESULTS.md"),"exports":[str(x) for x in exports]})
    print("WORKFLOW COMPLETE:",a.report_dir/"RESULTS.md",flush=True)
