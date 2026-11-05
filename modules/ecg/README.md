# ECG module

Dataset: **PTB-XL**. Intended task: **LVH-associated ECG pattern detection**.

This module is independent of the other signal pipelines. Algorithms, framework, architecture and windowing are not prescribed by the shared structure.

## Locations

| Path | Responsibility |
| --- | --- |
| `src/wearable_ecg/datasets/` | Public-dataset loaders and provenance |
| `src/wearable_ecg/validation/` | Signal/data integrity checks |
| `src/wearable_ecg/labels/` | Dataset-specific label mapping and exclusions |
| `src/wearable_ecg/windowing/` | Segmentation, timestamps and boundary rules |
| `src/wearable_ecg/preprocessing/` | Signal-specific fitted and runtime transformations |
| `src/wearable_ecg/models/` | Model definitions |
| `src/wearable_ecg/training/` | Training, objectives and checkpoint lifecycle |
| `src/wearable_ecg/evaluation/` | Offline, subject/patient and sensor-domain evaluation |
| `src/wearable_ecg/inference/` | Export loading, input validation and prediction interface |
| `configs/` | Independent dataset, preprocessing, model, training, evaluation and inference configs |
| `scripts/` | Prepare/train/evaluate/export entry-point stubs |
| `experiments/` | Versioned experiment configs and exploratory notebooks |
| `tests/` | Module unit/integration tests and synthetic fixtures |
| `docs/` | Dataset, pipeline and hardware-domain documentation |
| `environments/` | Research/inference dependency locks once implementations exist |

Dataset root: `dataset_roots.ptbxl` in repository `configs/paths.local.json`. Local working data: `data/ecg/`. Own checkpoints, exports, reports and runs: `artifacts/ecg/`. These payloads are ignored by Git.

## Non-negotiable data rules

Use the exact LVH diagnostic statement, not the HYP superclass. Metadata is not anatomical LVH confirmation. Keep all recordings of a patient in the official folds, including all derived windows. Freeze normal-control exclusions and confidence rules before training. Retain 12-lead research/teacher experiments separately from the single-lead wearable model.

Check waveform/metadata linkage, units, lead names, missingness, clipping, diagnosis contradictions, duplicate recordings and official patient-fold independence.

Keep split construction/auditing module-owned under `datasets/splits/`. Keep hardware-domain evaluation under `evaluation/hardware_domain/`. Shared helpers may check group overlap, but must not replace this module's split policy.

## Implementation status

Configs are templates. Null fields require explicit decisions and validation before use. Scripts and `ECGPredictor` intentionally raise an implementation-pending error. No training, inference, sensor acquisition or score calculation is implemented. Do not import training code from the inference package.
