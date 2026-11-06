# ECG: PTB-XL LVH-pattern research pipeline

Runnable independent ECG training/evaluation pipeline. The target is **LVH-associated ECG diagnostic statements**, not imaging-confirmed anatomical LVH. Other health modules remain separate.

## Completed comparison

The November 2026 comparison covers 30 supervised runs on two LVH annotation tasks. For the measured results, figures, verification evidence and open experiments, see the [research progress update](../../research/research_review_and_protocol.md#progress-as-of-6-november-2026) and [full results report](../../research/results/20261105_lvh_comparison/RESULTS.md).

## Run the experiment

From the repository root, create a Python 3.12 virtual environment and install `environments/research/requirements.txt`. The CUDA wheel is for the Windows research workstation, not Jetson Nano. The reproducible pipeline never downloads or writes to the source dataset.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r modules/ecg/environments/research/requirements.txt
.\.venv\Scripts\python.exe -m pip install -e packages/common -e modules/ecg
.\.venv\Scripts\python.exe -m pytest modules/ecg/tests -q
.\.venv\Scripts\python.exe modules/ecg/scripts/run_workflow.py `
  --source 'C:\Final Project\Datasets\ptb-xl-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' `
  --cache data/ecg/processed/ptbxl_360hz_v1 `
  --output artifacts/ecg/runs/20261105_lvh_comparison `
  --config modules/ecg/experiments/configs/lvh_comparison.json `
  --report-dir research/results/20261105_lvh_comparison
```

Source path is local to each teammate. In this workspace, `.venv/Scripts/python.exe` is the configured interpreter. Training takes substantial workstation time: 30 neural runs, each with development and refitting stages. Completed runs are resumable. Do not start two workflows against the same cache/output directory. An opened test comparison cannot be retrained in place.

## Pipeline and outputs

1. Verify official source SHA-256 checksums, calibrated mV units, lead names, 500-Hz records and patient-fold independence. Keep original files unchanged.
2. Freeze LVH-versus-strict-normal and LVH-versus-rest manifests. Exclude the LVH/NORM contradiction. A confidence of zero does not make a present LVH code negative.
3. Create 360-Hz Lead II and research-only 12-lead caches in `data/ecg/processed`. Fit the conservative correction controller on folds 1–6; assess its morphology proxies on folds 7–8.
4. Compare small CNN, CNN–Transformer and CNN–diagonal-SSM backbones with identical signal, feature, augmentation and training contracts. Five seeds per model/task. Separate physical-voltage bypass; no demographic or report inputs.
5. Select epochs on folds 7–8, refit on 1–8, calibrate and freeze thresholds on fold 9, then evaluate fold 10. Never select a seed or architecture from test scores.
6. Save learning curves, ROC/PR, calibration, confusion matrices, subgroup metrics, cluster bootstrap intervals, paired comparisons, predictions, checkpoints and environment/configuration records.
7. Export each architecture's prespecified seed 42 with preprocessing and calibration. Test fixed-shape ONNX parity on training windows. Rehash source files after training.

Full outputs live under ignored `artifacts/ecg/` and `data/ecg/`. The shareable aggregate report and graphs live under `research/results/`. Record-level metadata and predictions stay in ignored artifacts. Models are research artifacts; hardware-domain validation and physical Nano benchmarking remain necessary.

## Implementation map

- `datasets/prepare.py`: read-only WFDB ingestion, source checksums and caches.
- `labels/ptbxl.py`: exact statement cohorts and official patient folds.
- `preprocessing/pipeline.py`: minimal/conventional/SWT/controller variants and same-lead features.
- `models/backbones.py`: compact backbones, voltage bypass, reliability-biased attention.
- `training/runner.py`: development, refit, calibration and checkpoint lifecycle.
- `evaluation/`: independent preprocessing diagnostics, metrics, plots and patient-cluster uncertainty.
- `inference/`: strict calibrated-input contract, buffered prediction and ONNX export.
- `experiments/configs/lvh_comparison.json`: fully specified executed research experiment. The older `configs/*.json` files describe module-level integration contracts; their unchosen hardware fields remain null.

The state-space comparator is **not Mamba**. The preprocessing controller is an experimental realization with automatic fiducial proxies, not a validated novel clinical algorithm. Cross-lead distillation/pretraining and the publication-strength ablation program in `research/research_review_and_protocol.md` remain subsequent experiments; the supervised comparison does not silently stand in for them.
