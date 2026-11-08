# PTB-XL morphology distillation

The research endpoint is an LVH-associated SCP statement versus the existing eligible rest cohort. Models accept ten-second ECG windows. Twelve-lead signals are privileged training inputs. The exported predictor reads one calibrated Lead-II-equivalent waveform.

Start at the team repository root with the ECG environment and the prepared PTB-XL 1.0.3 cache. The scripts use the source and cache locations recorded during the original 360 Hz study.

~~~powershell
.\.venv\Scripts\python.exe modules/ecg/experiments/lvh_distillation/run.py
.\.venv\Scripts\python.exe modules/ecg/experiments/lvh_distillation/joint_ablation.py
.\.venv\Scripts\python.exe modules/ecg/experiments/lvh_distillation/distillation_comparison.py
.\.venv\Scripts\python.exe modules/ecg/experiments/lvh_distillation/audit.py
.\.venv\Scripts\python.exe modules/ecg/experiments/lvh_distillation/publish.py
~~~

E0 uses the original seed-42 runner. Its development AP tolerance is .005 and its sensitivity tolerance is .02 absolute. Each challenger uses the same seven-epoch training budget and mixed-precision CUDA path. Model selection uses patient-level AP on folds 7 and 8. Models are then refit on folds 1–8. Calibration and thresholds use fold 9, and final evaluation uses fold 10. All source ECGs remain in their local dataset directory.

The development-only joint experiment fits a model using every proposed supervision source, followed by matched component removals. The main study admits components separately for the combined E10 candidate. The fixed contrast analysis uses development scores to select the logit-distillation parameters and evaluates the planned student comparisons. This cohort's final fold has been studied before.

For a Jetson Orin Nano replay, pipe JSON lines with matching `counts` and `timestamps` arrays (seconds) to `bioamp_stream.py`. Supply the exported bundle, `--fs`, measured `--zero` and `--mv-per-count`, verified `--polarity`, and the actual ADC `--rail-min` and `--rail-max`. The stream emits a score every two seconds following ten continuous seconds. The analog front end connects to an MCU or external ADC. Device commissioning requires verified polarity, electrode placement, clock and gain.

[Results and figures](../../../../research/results/lvh_distill_v1/REPORT.md) · [Development table](../../../../research/results/lvh_distill_v1/development.csv) · [Joint ablations](../../../../research/results/lvh_distill_v1/joint_ablation.csv) · [Paired KD contrasts](../../../../research/results/lvh_distill_v1/paired_kd.csv)
