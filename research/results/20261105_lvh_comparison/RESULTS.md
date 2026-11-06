# PTB-XL Lead II LVH-pattern model comparison

This report contains measured results from the local PTB-XL 1.0.3 experiment. The target is the LVH ECG diagnostic statement, not imaging-confirmed anatomical LVH.

## Experimental design

500-Hz physical waveforms were read without modification and resampled to 360 Hz in a separate cache. Models receive a complete 10-second Lead II window and same-lead derived features. Official patient folds were retained: 1–6 for development training, 7–8 for epoch selection, 1–8 for refitting, 9 for probability calibration and thresholds, and 10 for one frozen final comparison. Five random seeds (42–46) were run per architecture and task. Class-weighted BCE, AdamW, batch size 32, a 100-epoch ceiling and patience 15 were shared. No demographics, text reports or diagnostic codes were model inputs.

CNN, CNN–Transformer and CNN–diagonal-SSM share a multiscale stem, separate physical-voltage branch and train-scaled same-lead morphology features. The SSM is a real diagonal time-invariant exponential-kernel model; it is not the input-selective Mamba architecture.

## Individual-model results

Values are mean ± sample standard deviation across five separately trained seeds. Thresholds are selected for at least 90% specificity on fold 9; test specificity is allowed to differ. Average precision is reported as AUPRC. Ensembles are separately labeled in their files and plots.

| Task | Architecture | AUROC | AUPRC | Sensitivity | Specificity | F1 | Brier | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lvh_vs_normal | cnn | 0.9329 ± 0.0044 | 0.8838 ± 0.0057 | 0.8224 ± 0.0148 | 0.9084 ± 0.0147 | 0.7752 ± 0.0130 | 0.0674 ± 0.0019 | 94,753 |
| lvh_vs_normal | cnn_ssm | 0.8997 ± 0.0081 | 0.8366 ± 0.0135 | 0.7626 ± 0.0261 | 0.9047 ± 0.0093 | 0.7351 ± 0.0166 | 0.0818 ± 0.0044 | 82,529 |
| lvh_vs_normal | cnn_transformer | 0.9102 ± 0.0030 | 0.8507 ± 0.0046 | 0.7710 ± 0.0136 | 0.9118 ± 0.0072 | 0.7487 ± 0.0066 | 0.0777 ± 0.0017 | 92,835 |
| lvh_vs_rest | cnn | 0.8105 ± 0.0060 | 0.4016 ± 0.0129 | 0.5196 ± 0.0084 | 0.8921 ± 0.0091 | 0.4128 ± 0.0114 | 0.0725 ± 0.0008 | 94,753 |
| lvh_vs_rest | cnn_ssm | 0.7925 ± 0.0071 | 0.3778 ± 0.0067 | 0.4748 ± 0.0102 | 0.8948 ± 0.0059 | 0.3877 ± 0.0129 | 0.0742 ± 0.0006 | 82,529 |
| lvh_vs_rest | cnn_transformer | 0.7872 ± 0.0086 | 0.3783 ± 0.0087 | 0.4766 ± 0.0282 | 0.8948 ± 0.0069 | 0.3885 ± 0.0110 | 0.0742 ± 0.0007 | 92,835 |

## Controls

Fixed-configuration voltage/morphology logistic regression and random forest, age/sex-only logistic regression, and a quality-only logistic control use the same cohort/folds. These are single deterministic control runs, not five-seed deep-learning estimates.

| Task | Control | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| lvh_vs_normal | voltage_logistic | 0.8188 | 0.6941 | 0.6121 | 0.9073 |
| lvh_vs_normal | voltage_random_forest | 0.8812 | 0.7505 | 0.6636 | 0.8987 |
| lvh_vs_normal | age_sex_logistic | 0.7831 | 0.5456 | 0.4626 | 0.9073 |
| lvh_vs_normal | quality_logistic | 0.4799 | 0.2286 | 0.0888 | 0.9087 |
| lvh_vs_rest | voltage_logistic | 0.7275 | 0.2745 | 0.3738 | 0.9012 |
| lvh_vs_rest | voltage_random_forest | 0.7468 | 0.2806 | 0.4019 | 0.8947 |
| lvh_vs_rest | age_sex_logistic | 0.6359 | 0.1308 | 0.1682 | 0.8805 |
| lvh_vs_rest | quality_logistic | 0.5027 | 0.0975 | 0.0000 | 1.0000 |

## Preprocessing evidence and scope

**Observed preprocessing outcome:** the fitted controller selected identity/minimal correction for every cached recording. Consequently, the neural architecture results are effectively a minimally processed, calibrated-Lead-II baseline comparison. Independent synthetic-pair diagnostics also found no improvement of this fitted controller over identity. This run does not establish the proposed denoising contribution. The conservative baseline and reported negative preprocessing result are retained without changing the frozen comparison.

The candidate uses a training-only decision tree to select identity, robust baseline correction, or bounded stationary-wavelet correction with conditional mains subtraction. Selection targets synthetic-pair reconstruction error subject to engineering amplitude, duration and ST proxy limits. Independent fold-7/8 fidelity results are saved, including failed budgets and clean-signal changes. Automatic landmarks are not a validated clinical delineator. The candidate is an initial implementation of the proposed hypothesis; neither novelty nor improved clinical morphology preservation has been established.

The controller may choose identity frequently. That is an observed limitation, not evidence that artifact removal is unnecessary. A separate per-record z-score cache supports a future classifier-level amplitude-removal ablation. Its dimensionless output is excluded from physical mV-error comparisons.

## Saved evidence

- `comparison_per_seed.csv` and `comparison_summary.csv`: every neural run and aggregate results.
- `plots/<task>/`: ROC/PR/reliability curves, confusion matrices, seed comparisons, learning curves, selective-risk curves and subgroup views, as PNG and PDF.
- Per-run `test/`: record-level predictions, patient-cluster bootstrap 95% intervals (1,000 resamples), subgroup metrics and desktop latency.
- Task directories: paired patient-bootstrap differences between five-seed ensembles. These intervals condition on trained models and do not replace training-seed variability.
- Each run: development/refit histories, checkpoint, configuration, calibration parameters, frozen thresholds and model checksum.
- `source_integrity_after.json`: post-experiment rehash of every source file used.
- Export bundles: PyTorch, fixed-shape ONNX, identical preprocessing controller, calibration, input contract and measured CPU ONNX parity.

## Interpretation limits and remaining protocol work

This completes the supervised architecture comparison stage. Cross-lead teachers and reliability-gated distillation, JEPA/pretraining, a full classifier-level preprocessing/KD ablation matrix, a clinically reviewed delineator, age/sex-matched sensitivity analysis, real-noise stress tests, and external Georgia/BioAmp validation are not claimed as completed experiments. No external pretrained checkpoint is used. These additions must have a separate frozen protocol; the present test results must not guide retuning on fold 10.

The original Nano, ADC calibration, sensor placement and real wearable-domain clinical/reference labels were not available for validation. Desktop latency excludes acquisition and preprocessing and does not establish Nano suitability. A 10-second buffered window incurs acquisition delay; it is not sample-causal inference. No anatomical LVH diagnosis or exclusion is supported.

## Local run locations

Full local outputs: `artifacts\ecg\runs\20261105_lvh_comparison`.

- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_normal\cnn`
- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_normal\cnn_transformer`
- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_normal\cnn_ssm`
- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_rest\cnn`
- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_rest\cnn_transformer`
- Export: `artifacts\ecg\exports\20261105_lvh_comparison\lvh_vs_rest\cnn_ssm`

## Development-based candidate selection

The following recommendations use mean internal-development sensitivity at the prespecified specificity, with average precision as tie-breaker. They were calculated from development histories, not from test rankings. All architectures and seeds remain reported. This is a research candidate choice; Nano timing and hardware validation may require a different backbone.

- lvh_vs_normal: **cnn**, internal-development sensitivity 0.8178, average precision 0.8557.
- lvh_vs_rest: **cnn_transformer**, internal-development sensitivity 0.5230, average precision 0.4126.

## Data and method attribution

Dataset: [PTB-XL 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), Wagner et al.; source data retain their supplied license. State-space background: [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893). The implemented real-exponential diagonal layer is an independent compact comparator, not a reproduction of the full S4D or Mamba systems.
