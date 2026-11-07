# OSASUD binary TCN + GRU follow-up

The two-time-scale hard-label model ranked respiratory-event seconds better than the original multitask TCN + GRU. Mean within-patient average precision (AP) increased from **.234 to .247**. It generated **17.7 fewer false events per scored hour** at its validation-selected threshold, while recovering **46.3%** of reference events against **62.4%** for the original model. Use the hard-label version as the binary research candidate. The original TCN + GRU remains the higher-recall reference at its event-F1 operating point.

## Four new fits per fold

The follow-up used the 30 original OSASUD patients, the frozen five-fold split, and exactly the same 821,545 scored seconds (228.21 h, 5,985 contiguous binary reference intervals). The source archive was read in the previous study. This experiment used its derived per-patient optical cache. Each model receives one optical stream, with every reference respiratory signal and subtype withheld from its inputs. [Original protocol and preprocessing](../../osasud_protocol.md)

| Model | Mean validation patient AP | Held-out patient AP | Event precision | Event recall | Event F1 | False events/h |
|---|---:|---:|---:|---:|---:|---:|
| Previous CNN + Transformer | .251 | .238 | .166 | .522 | .252 | 68.8 |
| Previous multitask TCN + GRU | .244 | .234 | .181 | .624 | .281 | 73.8 |
| New single-scale binary TCN + GRU | .250 | .242 | .181 | .576 | .276 | 68.2 |
| Two scales, hard one-second labels | .253 | .247 | .178 | .463 | .257 | 56.2 |
| Two scales, trailing fraction target | .241 | .239 | .190 | .476 | .272 | 53.2 |
| Two scales, hard labels and auxiliary fraction | .251 | .248 | .186 | .451 | .264 | 51.6 |

AP gives each patient equal weight. Event precision, recall and F1 pool interval matches across patients. Each model used a threshold fitted on the corresponding validation patients by mean patient event F1, five consecutive positive seconds to emit an event, and one-to-one temporal IoU ≥ .30. These thresholds yield different false-alarm rates. [Complete comparison](comparison.csv) · [Per-fold scores, thresholds and epochs](folds.csv) · [Model graph](models.png)

## What drove the change?

The **single-scale binary control** used the original 30 s TCN + GRU encoder with its five outputs replaced by one event logit per target second. It trained on the binary loss alone. Its .242 AP accounted for part of the improvement over the original multitask network.

The **two-time-scale network** used up to 12 s of pulse shape sampled at 16 positions per second. A causal waveform convolution extracted local pulse patterns. Mean and maximum activation within each second fed a short residual TCN. The longer branch received the 11 existing pulse/respiratory modulation features, four quality features and the history mask over as many as 90 s. Dilated temporal blocks (2, 4, 8) fed a 24-unit unidirectional GRU. The classifier combined the two branch outputs at each of the five target seconds. All preprocessing and convolutions were causal at the resolution of a completed input second. A clock discontinuity or invalid optical second restarted the history. Before 90 s had accrued, masked zero padding preserved the original 30 s warm-up and the original score coverage. The five scores still become available together at the five-second boundary.

For the **fraction target**, the label at second *t* was the proportion of the ten seconds ending at *t* labeled apnea or hypopnea. Fraction-only training minimized binary cross-entropy with this soft target. The auxiliary model used the original per-second binary loss plus 0.35 times a separate fraction-head loss. All calibrators, epoch scores, thresholds and held-out metrics used the original binary event labels. Supervision at boundaries therefore changed during training without changing the evaluation task.

Each fold used 18 training, six validation and six held-out patients. The four fits used matched training draws, fold-specific robust scalers, AdamW, up to ten epochs and the same early-stopping rule as the first study. Checkpoints were selected by within-patient validation AP. The per-fold logistic calibrator converted the candidate's primary logit to the probability of an event in the scored second. [Model implementation](../../../modules/ppg/src/wearable_ppg/models/dual_rate.py) · [Training and target construction](../../../modules/ppg/src/wearable_ppg/training/dual_rate.py)

## How much confidence to place in the differences

Relative to the original multitask TCN + GRU, the hard-label model gained **.0123 AP** (paired patient-bootstrap 95% interval +.0024 to +.0233). Against the matched binary control the gain narrowed to **.0045** (−.0043 to +.0123). The experiment shows a benefit from the combined binary and architectural changes. The added value attributable to two time scales alone remains uncertain.

The auxiliary head had the highest observed held-out AP of .248. Its paired gain over hard labels alone was .0014 (−.0029 to +.0055). Replacing hard labels with the fraction lowered mean AP by .0082 (−.0240 to +.0074), with higher event F1 at its selected threshold. The fraction-only model's matched-event median confirmation delay was 9 s versus 8 s for the other dual models. [Paired intervals](paired.csv)

The existing selection rule ranked the dual models by validation AP and used mean validation patient event F1 to break ties within .005 AP. It selected the **hard-label dual** for refit. Applied across all six fixed candidates it retained the **original CNN + Transformer**. With selection repeated independently inside each outer fold among the three dual candidates, the resulting patient AP was .242, event F1 .257, recall .450 and false events/h 53.7. The architecture was proposed after inspecting earlier results on this cohort. All performance estimates describe follow-up research on the same 30 patients. Intervals resample patients with checkpoints held fixed and exclude training-seed and external-site uncertainty. [Selection and fold choices](selection.json)

## Comparing alarm burdens

At a target of at most one false event per validation hour, the hard-label model detected **42 of 5,985** test events (0.70%) and incurred **1.06 false events per test hour**. The auxiliary model detected 68 (1.14%) at 1.77 false events/hour. These operating points have insufficient sensitivity for an unattended alert system.

In a secondary analysis we fitted thresholds for 1, 10 and 30 false events per validation hour, maximizing matched validation events under each allowance. Every encoder used the same threshold grid and alarm rule. At the 30/hour validation allowance the auxiliary model recovered **31.1%** of held-out events at **26.7** false events/hour. The original TCN + GRU recovered **27.0%** at **29.2** per hour. Actual false-alarm rates vary on held-out patients. A fixed, causal trailing five-second score average was also tried for each model, restarting at gaps. It raised the original TCN + GRU's event F1 from .281 to .291 and reduced its false events/h from 73.8 to 63.0 when thresholds were tuned for validation event F1. This was a decoder experiment using frozen predictions. The binary export uses the primary unsmoothed policy. [Operating-point graph](alert_budgets.png) · [All operating points](alert_policies.csv) · [Fold-specific choices](alert_policy_folds.csv)

## Research bundle and verification

The hard-label model has **9,097 parameters**. The refit used all 30 patients, nine epochs (the fold median), all-patient scaling, and a calibrator and threshold fitted to out-of-fold scores. Performance figures above come from held-out checkpoints. The ONNX graph is 71,673 bytes and consumes a batch of one 90 × 32 block including the real-history mask. In 128 real comparison blocks its maximum logit difference from PyTorch was 0.0000020. Incremental replay on three real, contiguous traces reproduced offline probabilities within 0.00000006, including the transition from padded to full history. Physical MAX30102 accuracy, Jetson latency and performance in an independent cohort remain unmeasured. [Export verification](export.json) · [Binary ONNX bundle](../../../artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard/)

The checkpoint verifier reconstructed each fold's training scaler and every timestamp and label in **40** validation and test prediction files, recomputed patient AP and replayed **61,040** blocks through **20** saved checkpoints. The maximum replayed logit difference was zero on the training GPU. All 12 PPG tests passed, including timing, filter resets, trailing-target boundaries and per-second causality.

From the repository root, use the existing PPG environment and the prepared v1 cache. The training command resumes completed folds. The results, graphs and selection files are in this report's directory. Per-fold weights, scalers, predictions and training histories are under `artifacts/ppg/runs/osasud_binary_dual_rate_v1/<variant>/fold<k>/`.

~~~powershell
.\.venv\Scripts\python.exe modules/ppg/scripts/train_binary_dual_rate.py
.\.venv\Scripts\python.exe modules/ppg/scripts/evaluate_binary_dual_rate.py
.\.venv\Scripts\python.exe modules/ppg/scripts/compare_binary_alerts.py
.\.venv\Scripts\python.exe modules/ppg/scripts/verify_binary_dual_rate.py
.\.venv\Scripts\python.exe modules/ppg/scripts/export_binary_dual_rate.py
.\.venv\Scripts\python.exe -m pytest modules/ppg/tests -q
~~~
