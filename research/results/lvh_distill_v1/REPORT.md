# Lead-II LVH-associated ECG-pattern study

The frozen development rule selected **prep_minimal** for single-channel inference and **E2_inception** as the privileged teacher. The forward-selected combination was **E3**. The endpoint is the PTB-XL LVH statement against eligible remaining ECGs. Repeated ECGs for each patient use maximum logit and an any-positive reference. The original seed-42 CNN reproduced development AP and sensitivity exactly.

## Official fold 10

Thresholds target 90% specificity on fold 9. These are achieved patient-level results on fold 10. This previously studied fold provides an internal follow-up comparison.

| name | auprc | auroc | sensitivity | specificity | f1 | ppv | npv | accuracy | balanced_accuracy | mcc | brier | ece_10bins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_inception | 0.7342 | 0.948 | 0.8564 | 0.896 | 0.6268 | 0.4943 | 0.9813 | 0.8918 | 0.8762 | 0.5982 | 0.0493 | 0.0142 |
| E3 | 0.4373 | 0.804 | 0.5149 | 0.8948 | 0.4289 | 0.3675 | 0.9395 | 0.8545 | 0.7048 | 0.3547 | 0.0765 | 0.0093 |
| prep_minimal | 0.4373 | 0.804 | 0.5149 | 0.8948 | 0.4289 | 0.3675 | 0.9395 | 0.8545 | 0.7048 | 0.3547 | 0.0765 | 0.0093 |
| E0 | 0.4295 | 0.8093 | 0.5248 | 0.886 | 0.4223 | 0.3533 | 0.9401 | 0.8477 | 0.7054 | 0.3472 | 0.0772 | 0.0127 |

[All patient operating points and record-level metrics](final.csv) · [Development ranking](development.csv) · [Ablation figure](ablation.png)

## Training and selection

One seed per challenger, seven complete training epochs with AdamW (0.001 learning rate and weight decay), batch size up to 128, gradient norm limit 2. E0 retained its original optimizer and epoch policy. Folds 1–6 train each challenger, 7 and 8 supply the mean patient AP. Each development fold's Brier uses calibration fitted on the other development fold. Specificity thresholds for the development tie-break are empirical ROC operating points. For refit, folds 1–8 train weights and fold 9 fits positive-slope logistic calibration and the operating thresholds. No representation learning uses folds 9–10. The equivalence margin is .005 AP, with sensitivity at 90% specificity, F1, size and amortized batch inference time as tie-breakers. The Brier tolerance is E0 + .01 and the quality-input permutation AP loss limit is .015 for fusion challengers.

E1 uses a deeper residual multiscale encoder with physical-voltage features. E2 compares plain channel concatenation, residual Inception-style processing and shared lead patches with identity embeddings and cross-lead attention. All teachers have masked statement losses and random lead dropout while always retaining II. E3 uses the plain single-channel residual family. E4 tests temperature 1 or 3 and KD coefficient .2 or .6. E5 adds cosine representation alignment (.15) and compares confidence weighting with its removal. The teacher gate uses twice the distance of the calibrated teacher probability from .5. Teacher logits for training students are in-sample, temperature-scaled Bernoulli targets. This can transmit teacher overconfidence. Both latent vectors are 48-dimensional. No patient demographics enter the networks.

E6 compares hard labels, positive-example likelihood weighting, likelihood/100 soft targets, and training with positive likelihood cutoffs 50 or 80. Negatives retain unit weight and zero target. The 100-likelihood validation analysis is a sensitivity subset. The original LVH/NORM contradiction is excluded, and there were no present LVH codes with unknown zero likelihood in this release. Filtering by annotation confidence changes case mix and is unavailable on wearable inputs. [Exact fold counts](confidence_counts.csv) · [Coverage versus performance](confidence_coverage.csv)

E7 tests ST–T, conduction, other hypertrophy and infarction groups separately. Both LVH and VCLVH are excluded from the group definitions. Absent group statements are annotation negatives; present statements with likelihood below 50 are masked. Each auxiliary loss has coefficient .2. E8 uses two development-training-only masked patch reconstruction epochs, MSE on the hidden 25% of patches. E9 combines raw waveform, median aligned beat, training-standardized voltage measurements and a quality gate. Beat-only, feature-only and fixed-gate variants retain the raw branch. Landmarks, QRS spans, ST and T measurements are automatic estimates. E10 only admits individually improving components, then tests combined configurations using the frozen development rule. Each checkpoint directory records its effective configuration and training losses.

Imbalance comparisons use positive-class BCE weighting, focal gamma 2, effective-number focal with beta .9999, log-odds adjusted BCE, balanced random sampling and a score-weighted sampler for difficult eligible negatives. All use the same patient partitions. Minimal, drift-corrected, fixed band-pass, conditional mains removal, training-fitted robust scaling and the original constrained preprocessing were compared. The 100 Hz experiment anti-aliases the calibrated 500 Hz source directly to 100 Hz and trains at that resolution. Synthetic gain, noise, offset, clock jitter, shifts, quiet-segment holds, quantization and clipping were tested as one augmentation package. Clean and perturbed development AP appear in the development table. The clipping audit uses extrema dwell as a proxy because original ADC rails are unavailable. [Waveform fidelity](preprocessing.csv) · [Shortcut-only probes](shortcut_probes.csv)

## Wearable contract

A three-electrode Lead-II approximation requires a verified RA negative and LL positive orientation and a reference electrode. The BioAmp EXG Pill analog output goes through an MCU or dedicated ADC. Timestamped counts, measured offset, verified polarity and calibrated input-referred mV/count are required. The Jetson Orin Nano receives digital samples. The standalone predictor uses a ten-second buffered window, accepts zero missing samples, and withholds scores on clock gaps, clipping or flat input. The bundle includes its fold-9 threshold, calibration, a fixed-shape ONNX student and any required fitted controller. The preprocessing implementation and serialized parameters determine the exact filters. The default polyphase FIR uses a Kaiser window with beta 5. Physical hardware sensitivity, power and latency await a paired device study. [Raw-input parity](export.json) · [Local student bundle](../../../artifacts/ecg/exports/lvh_distill_v1/prep_minimal/)

Run the study from the repository root with the existing ECG virtual environment, first running modules/ecg/experiments/lvh_distillation/run.py and then publish.py. Individual phase markers permit resumption of this exact experiment. Raw waveforms and patient-level outputs stay local. [PTB-XL release and SCP likelihood semantics](https://physionet.org/content/ptb-xl/1.0.3/)

## Distillation and component evidence

The selected student (prep_minimal) achieved patient AP 0.437, compared with 0.429 for the reproduced CNN and 0.734 for the 12-lead teacher. All three use the same eligible patients. The teacher–student difference measures this architecture and cohort; it is not an information-theoretic limit.

| Patient AP contrast on fold 10 | Difference | Paired 95% interval |
|---|---:|---:|
| E4_T1_w0.6 minus E3 | -0.012 | [-0.027, +0.002] |
| E5_repr minus E4_T1_w0.6 | -0.002 | [-0.020, +0.018] |
| E5_conf_gate minus E3 | -0.011 | [-0.028, +0.006] |
| joint_full minus joint_no_teacher | -0.010 | [-0.027, +0.010] |
| joint_full minus E3 | -0.029 | [-0.057, +0.002] |
| E2_inception minus prep_minimal | +0.297 | [+0.233, +0.357] |
| prep_minimal minus E0 | +0.008 | [-0.016, +0.034] |

Intervals use 1,000 paired patient bootstraps with frozen checkpoints. Each challenger was trained with one seed. Sampling intervals exclude training-seed variability. [KD final metrics at four operating points](kd_final.csv) · [Paired contrasts](paired_kd.csv)

The full joint morphology and teaching model reached development AP 0.434. Removing all teacher supervision gave 0.436; removing the explicit morphology fusion path gave 0.420. Interpret these as interactions within this loss and backbone. [Joint component removals](joint_ablation.csv) · [Effective model configurations](configurations.json)

Confidence filtering at likelihood 100 retained 95.2% of development records and 217 positive records. On the full source-window audit 0 recordings contained nonfinite samples, 0 were entirely flat, 31 exceeded ±5 mV, and 21795 had a constant stretch of at least 50 samples. A constant stretch can occur at the stored record boundary. This flag alone cannot establish ADC clipping. [Signal audit](signal_integrity.csv) · [Patient split audit](split_audit.json)

The wearable ten-second score uses separate fold-9 record-level calibration. At its frozen threshold, fold-10 window sensitivity was 0.491, specificity 0.896, and Brier 0.073. The reported raw-input parity checked 59 source records. [Window metrics](wearable_window_metrics.json)

The mixed-precision challenger runs are under development_amp; earlier full-precision timing pilots are excluded. Ranking uses one fixed budget across challengers. The greedy E10 search tests selected combinations and leaves untested interactions. Synthetic perturbations and retrospective clinical waveforms leave wearable electrode placement, gain, artifact and population shifts unmeasured.

[Held-out ROC, precision–recall and calibration plots](heldout_curves.png) · [BioAmp board reference](https://docs.upsidedownlabs.tech/hardware/bioamp/bioamp-exg-pill/index.html)
