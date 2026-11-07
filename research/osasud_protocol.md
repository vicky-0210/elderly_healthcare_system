# OSASUD PPG respiratory-event study

The primary endpoint is the physician's one-second annotation of any apnea or hypopnea. The secondary endpoint retains NONE, HYPOPNEA, APNEA-CENTRAL, APNEA-OBSTRUCTIVE and APNEA-MIXED. Evaluation includes five held-out groups of six patients each. Subject identity is available to the splitter and evaluator only. [OSASUD data descriptor](https://doi.org/10.1038/s41597-022-01272-y)

## What the optical waveform can contribute

OSASUD synchronizes Mindray monitoring with respiratory annotations from overnight stroke-unit recordings. The distributed plethysmogram is one 80 Hz monitor waveform. We exclude ECG, SpO₂, respiratory rate, nasal airflow, effort belts, the source anomaly flag and patient metadata from model inputs. Physician labels supervise development. The monitor's gain control, spectral response and wavelength mixing are unknown. The MAX30102 produces separate reflectance red and infrared ADC streams, so the initial deployment uses a single chosen optical channel and requires paired-domain validation.

Pulse amplitude, interval, width and upslope variability have prior use in apnea sensing. Pulse rate and intensity modulation can lag a respiratory disturbance, and peripheral perfusion and motion can also affect them. Published OSASUD oximetry Transformer results use oxygen-saturation information and a different input/endpoint. This study compares PPG-only models on matched participants and times. [Pulse waveform study](https://ieeexplore.ieee.org/abstract/document/8743678) · [ApSense](https://arxiv.org/abs/2306.10863) · [OSASUD oximetry study](https://doi.org/10.3390/s23187924)

## Reconstruction and features

Sort each patient's one-second packets by timestamp. Reject both copies of duplicate timestamps, incomplete/nonfinite packets and whole-second flatlines. Split when there is a timestamp discontinuity or invalid signal. Restart all filters and pulse histories after a gap. The checksum-identified release layout can be read in bounded memory with a checksum-locked static projection of its pickle opcodes. Source indices, timestamps and five-class labels travel with each derived patient file. Estimates for unusable periods stay unavailable.

The method under test combines three views. First, shrink only the finest Haar details with a robust within-second threshold and preserve each pair's mean. Filter the cardiac component at 0.45–5.5 Hz and retain 16 equally spaced pulse-shape values per second. Second, estimate relative baseline shift, log pulse-range modulation, pulse intervals from completed upward crossings, band-limited modulation and their trailing energies. The respiratory proxy band is 0.08–0.38 Hz (4.8–22.8 cycles/min). Direct relative-amplitude and interval channels retain changes outside that band. Third, keep lagged low-pass residual, repeated-sample fraction, beat staleness and dwell at observed extrema. The last measure flags plateau-like behavior; the Mindray ADC rail is unknown. Fit global median/IQR features on training patients only and apply asinh compression. A 45 s exponentially weighted reference tracks local sensor gain without erasing all between-window amplitude variation. Every transform uses data available at the end of its input second.

## Four primary models

| Candidate | Mechanism |
|---|---|
| Multiscale CNN + Transformer | Short/long temporal kernels, sinusoidal positions and self-attention over completed context. |
| Dilated TCN + GRU | Causal separable temporal blocks feeding a recurrent state. |
| Multiresolution filter bank + selective state space | Separate short/medium changes and a slow trend, followed by a learned input-dependent diagonal recurrence. This is a compact state-space implementation. |
| Quality-conditioned dual TCN + cross-attention | Pulse and modulation encoders, queries from the last five modulation tokens into the pulse history, then a learned featurewise gate informed by signal quality. |

Use 30-second windows every five seconds. Predict the five individual event labels at the end of each window with an event logit and a conditional four-way subtype head. Thus a window can straddle onset, offset or changes of subtype. Output for a labeled second becomes available at the block end, 0–4 seconds after that second has finished. The separate lag experiment attributes the five scores to seconds ten seconds earlier and incurs an additional ten seconds of delay. The first complete window is required after every stream reset. The 10 s and 20 s context controls use the same eligible endpoints as the 30 s model.

## Evaluation and selection

Sort patients by scorable event prevalence, shuffle within consecutive blocks of five (seed 2729), and assign one from each block to each fold. Each fold trains on 18 patients, validates on six and tests on six. Choose the epoch by validation macro-patient average precision (AP). Keep that very checkpoint to preserve its score scale. Fit logistic calibration on validation seconds. Choose a threshold from 26 validation-score quantiles by mean patient event F1. Restore population prevalence in validation and testing after sampling a mixture of uniform and event-containing training windows. Use AdamW, gradient clipping, up to ten epochs, early stopping after three epochs without improvement, a binary loss plus weighted conditional subtype cross-entropy.

Rank the four primary architectures by their mean inner patient AP. Among models within 0.005 AP, prefer the higher validation patient event F1, then fewer parameters. Apply the rule independently within each outer fold to estimate selection performance. Report the fixed-model comparisons separately. The waveform-only, modulation-only, quality-only, fixed-gate, context length and delayed-output controls test specific choices and are excluded from selection.

A reference event is a contiguous run of positive physician seconds. A predicted event needs five consecutive above-threshold seconds. Match predicted and reference intervals one-to-one at temporal IoU ≥0.30 in descending IoU order. Count unmatched predictions as false events, with all scored hours in the denominator. Report event coverage, per-second discrimination, five-class confusion and confirmation timing on matched runs. A clinical event index per hour of sleep requires validated sleep time. The algorithm estimates neither sleep duration nor a clinical diagnosis.

The novelty hypothesis is the joint use of a mean-preserving noise path, matched pulse and respiration time scales, quality-conditioned cross-view fusion and boundary-resolving supervision. Each component has precedents. The controls test their contribution on this cohort; a first-in-literature claim would require a broader systematic review.
