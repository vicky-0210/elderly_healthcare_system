# WESAD wrist cohort and window audit

The primary endpoint is classification of the WESAD baseline and laboratory stress protocol conditions from the wrist. A positive prediction means stress-associated physiological arousal. Participants were 24–35 years old. Elderly users, daily living, and the proposed GSR hardware need separate validation.

## Inputs and integrity

The loader opens the trusted local WESAD subject pickles in read-only mode. It selects the synchronized wrist arrays and the original 700 Hz protocol annotation. It reads the released E4 CSV headers to check sample rates and initial timestamps. The synchronized pickle has a relative common origin. Using the raw CSV arrays in its place would require repeating the synchronization in the release. Pickle loading executes Python deserialization and is intended only for the trusted release.

The extraction and later independent reread produced matching SHA-256 checksums for all 15 source pickles. The local manifest under `data/eda/processed/wesad_v1/manifest.json` contains sizes, timestamps, source hashes and calibration boundaries. Preparation skips complete outputs with matching source size and modification time. The audit checks content hashes separately.

| Released evidence | Observed in this copy |
| --- | --- |
| Participants | 15, S2–S11 and S13–S17 |
| Age, mean ± sample SD | 27.47 ± 2.45 years |
| Recorded gender | 12 male, 3 female |
| Synchronized duration | 24.13 h total, 87.05–117.85 min per person |
| EDA and temperature | 4 Hz |
| Accelerometer and BVP | 32 Hz and 64 Hz |
| Original label clock | 700 Hz, exactly 175 annotations per quarter-second |
| Wrist streams with unequal start timestamps or trimmed tails | 0 |
| Nonfinite wrist entries in synchronized arrays | 0 |
| Observed wrist EDA | 0.045–15.921 µS |

The four raw wrist start timestamps agreed within each subject. All four synchronized arrays covered exactly the released annotation duration. These checks establish compatible durations and origins. The release does not provide an independent physiological event marker that would establish sub-sample alignment or clock drift at every point.

## Class assignment and folds

Every 4 Hz tick stores counts from its original 175 annotation samples. The loader totals these counts over a candidate window, including all released codes in the denominator. Only a requested target code occupying at least the specified fraction can label a window. The default 80% policy allows up to 20% other context at a boundary. A 100% comparison removes mixed windows. Codes 0 and 5–7 cannot become a supervised target. The primary target codes are 1 (baseline) and 2 (stress). Secondary targets add 3 (amusement) and 4 (meditation).

The system reserves the first complete 120-second run of resting code 1 per participant for explicit calibration. Both research protocols start scoring after that interval, on a common 15-second grid. Calibration samples and earlier observations cannot form scored windows. The zero-calibration path uses the shared schedule for a matched comparison and takes no personal normalization statistics.

The baseline 60-second, 15-second-stride design produced 1,008 baseline and 628 stress windows. Participants contributed 65–69 baseline and 39–46 stress windows each. Overlap is 75%. Precision estimates therefore use participants as the independent units. A subject owns each window before pooled feature arrays are formed. No cross-participant operation occurs during segmentation. The model runners hold out the entire subject before any learned transform or model fit.

| Protocol code | Condition | Minutes before window restrictions |
| --- | --- | ---: |
| 0 | Undefined or transition | 658.45 |
| 1 | Baseline | 293.52 |
| 2 | Stress | 166.10 |
| 3 | Amusement | 92.92 |
| 4 | Meditation | 196.77 |
| 5–7 | Other non-target annotations | 40.05 |

## Reproduce

Run from the repository root with the EDA research dependencies installed. On PowerShell:

```powershell
$env:PYTHONPATH = 'modules/eda/src'
python modules/eda/scripts/prepare_wesad.py --source 'C:/Final Project/Datasets/wesad'
python modules/eda/scripts/audit_wesad.py --source 'C:/Final Project/Datasets/wesad' --publish research/results/wesad_loso_v1/preprocessing
python -m unittest discover -s modules/eda/tests/unit -p test_wesad_signals.py -v
```

The portable results include the [recording-level audit](../../../research/results/wesad_loso_v1/preprocessing/recording_audit.csv), [dataset summary](../../../research/results/wesad_loso_v1/preprocessing/audit_summary.json), [condition durations](../../../research/results/wesad_loso_v1/preprocessing/cohort_conditions.png) and [representative four-condition wrist traces](../../../research/results/wesad_loso_v1/preprocessing/protocol_examples.png). Source information is available from the [WESAD maintainers](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) and [Schmidt et al.](https://doi.org/10.1145/3242969.3242985).
