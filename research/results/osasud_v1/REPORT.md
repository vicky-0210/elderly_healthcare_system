# OSASUD PPG respiratory-event results

The multiscale CNN + Transformer ranked first on the validation criterion. Its held-out mean patient average precision (AP) was **0.238**, compared with **0.160** for a constant score within each patient. The TCN + GRU recovered the most events at the event-F1 operating point. Both generated dozens of false events per hour. The current export is a research replay model. The measured trade-off does not support a caregiver alert or identification of apnea subtype.

The subsequent binary study added a matched TCN + GRU control and three two-time-scale variants on the same 821,545 seconds. The new hard-label model reached .247 patient AP with fewer false events and lower event recall than the original TCN + GRU. [Follow-up comparison, operating points and export](../osasud_binary_dual_rate_v1/REPORT.md)

## Patient-independent model comparison

All four candidates saw the same optical waveform, supervision, training budget and outer test patients. AP averages the 30 within-patient precision–recall summaries equally. Event F1 pools one-to-one interval matches across scored hours. Five-class macro F1 gives each physician label equal weight.

| Architecture | Parameters | Validation patient AP | Held-out patient AP | Event precision | Event recall | Event F1 | False events/hour | Five-class macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale CNN + Transformer | 25,573 | .251 | **.238** | .166 | .522 | .252 | **68.8** | **.193** |
| Dilated TCN + GRU | 15,557 | .244 | .234 | **.181** | **.624** | **.281** | 73.8 | .191 |
| Multiresolution filter bank + selective state-space model | 14,053 | .236 | .228 | .171 | .603 | .266 | 76.9 | .182 |
| Quality-conditioned dual TCN + cross-attention | 17,861 | .234 | .229 | .166 | .601 | .260 | 79.1 | .179 |

Parameters count the instantiated networks, including inactive weights where applicable. The CNN + Transformer scored above the constant baseline in 26 of 30 patients. Patient AP ranged from .005 to .731. A patient bootstrap gave a 95% interval of .170–.306 for the mean. Its paired AP difference against TCN + GRU was .0031 (95% interval −.0048 to .0122). These are descriptive intervals over this cohort with fixed fold predictions. They exclude retraining and external-site variation. [Comparison](comparison.csv) · [Paired differences](paired_ap.csv) · [Fold metrics](fold_metrics.csv) · [Model comparison figure](figures/models.png)

The fixed architecture for export was chosen by mean inner-validation AP, with a .005 tolerance for event-F1 tie-breaking. Independently repeating the choice within each outer fold picked CNN + Transformer three times and TCN + GRU twice. The resulting held-out estimates were .238 patient AP, .251 event F1 and 70.4 false events/hour. These estimates describe the architecture-selection procedure. All-patient refit performance requires external evaluation. [Selection record](selection.json) · [Training curves](figures/training.png)

## Source, coverage and timing

Thirty overnight stroke-unit recordings contributed 961,357 labeled seconds. We read the 80 Hz Mindray waveform samples and physician labels from the local original. For each patient we retained source row indices, sample times and event type. No other physiological stream entered the models. The release had 130,309 seconds with nonfinite PPG and 580 further flat seconds. The one-second timestamps had no duplicates, gaps or reversals within a patient. Missing optical samples inside timestamped packets account for most of the coverage loss. [Source audit](source_audit.json) · [Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC9018698/)

The frontend reset across unusable packets. From 830,468 usable seconds, warm-up and incomplete terminal blocks left **821,545 scored seconds (228.21 h, 85.46% of the archive)**, including 131,085 positive seconds. All four models used 164,309 overlapping 30 s contexts at a 5 s stride. Each context predicted its final five individual seconds, so each eligible second was scored exactly once. Of 6,608 original contiguous positive annotations, 5,984 received some coverage and 5,926 received complete coverage. One or more interruptions can split a source event into multiple scored fragments. This yielded 5,985 reference intervals for the primary comparisons. Consecutive positive seconds of different subtypes form one binary interval.

Each of five folds used 18 training patients, six validation patients and six held-out patients. We fit the scaler on training data, selected epochs and fitted calibration and thresholds on validation data, and evaluated untouched test participants. Each training epoch sampled 400 uniform and 240 event-enriched contexts per training patient, with a uniform fallback when no positive context was present. Evaluation kept the observed prevalence. Each scored second was available at its block boundary, zero to four seconds after that second ended. This is retrospective labeling over a short block. [Method, label definitions and selection](../../osasud_protocol.md)

## Alert burden and subtype errors

At the primary threshold the selected model matched 3,125 intervals, missed 2,860 and generated 15,707 unmatched predictions. A predicted interval required five consecutive positive seconds and temporal intersection over union of at least .30 for a one-to-one match. Median confirmation delay among matched intervals was six seconds from annotated onset. Second-level precision was .191, recall .582, specificity .532, pooled AP .202 and AUROC .578. [Precision–recall and calibration plots](figures/pr_calibration.png)

An exploratory operating point maximized matched validation events within an allowance of one false event per validation hour. We applied those thresholds unchanged to the corresponding held-out patients. Only **24 of 5,985** events were matched (**0.40% recall**) with **263 false events, or 1.15 per hour**. The original threshold and export are unchanged by this follow-up. [Alarm-burden experiment](alert_burden.csv)

| Label | Scored seconds | Patients with label | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| NONE | 690,460 | 30 | .870 | .532 | .660 |
| HYPOPNEA | 69,699 | 30 | .101 | .410 | .162 |
| APNEA-CENTRAL | 10,268 | 22 | .028 | .030 | .029 |
| APNEA-OBSTRUCTIVE | 39,374 | 27 | .076 | .177 | .106 |
| APNEA-MIXED | 11,744 | 14 | .008 | .008 | .008 |

The hierarchy targets the binary event first and then the conditional subtype. A single plethysmogram provides indirect information about respiratory changes. The observed central and mixed apnea recalls make those outputs unsuitable for distinction in practice. [Class metrics](class_metrics.csv) · [Confusion matrix](figures/subtypes.png)

## Which proposed changes helped?

The research contribution tested here combines within-second, mean-preserving Haar shrinkage, pulse shape and slower respiratory modulation views, signal-quality features and separate labels at event boundaries. The most consistent gain came from retaining modulation information. The structurally matched dual-stream model reached .229 patient AP. Masking modulation and quality to retain morphology alone reduced AP to .188. The paired difference was +.041 (95% interval +.023 to +.060). Modulation alone reached .223, while quality alone reached .165. Adding morphology and quality to modulation alone gave +.006 (−.002 to +.014). This design cannot isolate each input feature's individual causal contribution. [Ablation plot](figures/ablations.png) · [Paired comparisons](paired_ap.csv)

A learned fusion gate reached .229 AP against .228 for a fixed half-and-half gate. The paired difference was +.0005 (−.0034 to +.0047). Matched 10 s, 20 s and 30 s contexts reached .220, .224 and .229. A separate model permitted ten more seconds of observation after its targets and reached .243 AP and .274 event F1. Its matched-event delay rose to 16 s. The delayed task shifts target coverage slightly at run boundaries. These findings support preserving slow information and testing the allowed reporting delay. They do not establish a general advantage for adaptive attention, Haar shrinkage or the state-space model. [All controls](comparison.csv)

## Export and MAX30102 transfer

The final CNN + Transformer was refit on all 30 people for the median selected epoch count of two. Its bundle includes trained weights, a static ONNX graph and the exact feature list, normalization parameters, calibration and threshold. Refit calibration and threshold used out-of-fold logits. The export measured 137,245 bytes. On 128 comparison blocks its maximum logit difference from PyTorch was 7.2 × 10⁻⁷. On three contiguous real traces, incremental 137-sample packets reproduced offline probabilities to the recorded precision. Host CPU model-only p50/p95 were 0.114/0.126 ms, excluding acquisition, feature extraction and the five-second cadence. [Verification record](export_verification.json) · [Local bundle](../../../artifacts/ppg/exports/osasud_v1/cnn_transformer/)

Frozen outer checkpoints also processed a label-blind central ten-minute excerpt from the longest continuous section of each recording. Across 17,250 scored seconds, disabling Haar shrinkage changed 16 threshold decisions. Simulated 80-to-100 Hz interpolation changed 716 (4.2%). Adding gain and offset at 100 Hz also changed 716 relative to the original. Reversing polarity at 100 Hz changed 3,268 (18.9%) and the contact-noise simulation changed 2,945 (17.1%). Simulated dropouts left 11,700 comparable seconds after restart and warm-up. Excerpt scores have a different case mix from the full study. These perturbations model selected effects and give no estimate of accuracy on physical hardware. [Transfer table](transfer.csv)

The driver supports 100 Hz red/IR FIFO acquisition. The model consumes one channel at a time. Simultaneous wearable and respiratory-reference recordings are needed to establish site, polarity, channel selection, end-to-end latency and patient-level generalization. The dataset comes from a single stroke-unit population. Its scored duration includes recording time with usable PPG and cannot establish an apnea–hypopnea index per hour of confirmed sleep. [Hardware procedure](../../../modules/ppg/docs/hardware_validation.md) · [MAX30102 specifications](https://www.analog.com/en/products/max30102.html)

## Reproduce and verify

Run the PPG checkpoint verifier from the repository root. The completed audit verified all 55 checkpoints, all 110 held-out/validation prediction files, group membership, recalculated training scalers, reconstructed score timestamps and the original archive SHA-256. Replaying three native inference batches from each validation and test file reproduced the stored event logits exactly on the training workstation GPU. Six module tests passed. Model weights, per-second scores and per-patient manifests are local files. [Commands and output paths](../../../modules/ppg/README.md)
