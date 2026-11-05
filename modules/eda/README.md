# EDA/GSR module

Dataset: **WESAD**. Intended task: **Laboratory baseline versus stress detection**.

This module is independent of the other signal pipelines. Algorithms, framework, architecture and windowing are not prescribed by the shared structure.

## Locations

| Path | Responsibility |
| --- | --- |
| `src/wearable_eda/datasets/` | Public-dataset loaders and provenance |
| `src/wearable_eda/validation/` | Signal/data integrity checks |
| `src/wearable_eda/labels/` | Dataset-specific label mapping and exclusions |
| `src/wearable_eda/windowing/` | Segmentation, timestamps and boundary rules |
| `src/wearable_eda/preprocessing/` | Signal-specific fitted and runtime transformations |
| `src/wearable_eda/models/` | Model definitions |
| `src/wearable_eda/training/` | Training, objectives and checkpoint lifecycle |
| `src/wearable_eda/evaluation/` | Offline, subject/patient and sensor-domain evaluation |
| `src/wearable_eda/inference/` | Export loading, input validation and prediction interface |
| `configs/` | Independent dataset, preprocessing, model, training, evaluation and inference configs |
| `scripts/` | Prepare/train/evaluate/export entry-point stubs |
| `experiments/` | Versioned experiment configs and exploratory notebooks |
| `tests/` | Module unit/integration tests and synthetic fixtures |
| `docs/` | Dataset, pipeline and hardware-domain documentation |
| `environments/` | Research/inference dependency locks once implementations exist |

Dataset root: `dataset_roots.wesad` in repository `configs/paths.local.json`. Local working data: `data/eda/`. Own checkpoints, exports, reports and runs: `artifacts/eda/`. These payloads are ignored by Git.

## Non-negotiable data rules

Use wrist EDA from Empatica E4 at 4 Hz, not a chest signal with a different sample rate. Keep source labels 0..4, using only baseline (1) and stress (2) for the initial binary task. Align the annotation clock to the wrist signal; do not assume identical array rates or lengths.

Check subject identity, wrist stream selection, 4 Hz timing, annotation alignment, included/excluded states, gaps, saturation and subject-independent splits. Do not describe laboratory label classification as clinical stress diagnosis.

Keep split construction/auditing module-owned under `datasets/splits/`. Keep hardware-domain evaluation under `evaluation/hardware_domain/`. Shared helpers may check group overlap, but must not replace this module's split policy.

## Implementation status

Configs are templates. Null fields require explicit decisions and validation before use. Scripts and `EDAPredictor` intentionally raise an implementation-pending error. No training, inference, sensor acquisition or score calculation is implemented. Do not import training code from the inference package.
