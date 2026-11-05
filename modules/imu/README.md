# IMU module

Dataset: **SisFall**. Intended task: **Fall versus ADL detection**.

This module is independent of the other signal pipelines. Algorithms, framework, architecture and windowing are not prescribed by the shared structure.

## Locations

| Path | Responsibility |
| --- | --- |
| `src/wearable_imu/datasets/` | Public-dataset loaders and provenance |
| `src/wearable_imu/validation/` | Signal/data integrity checks |
| `src/wearable_imu/labels/` | Dataset-specific label mapping and exclusions |
| `src/wearable_imu/windowing/` | Segmentation, timestamps and boundary rules |
| `src/wearable_imu/preprocessing/` | Signal-specific fitted and runtime transformations |
| `src/wearable_imu/models/` | Model definitions |
| `src/wearable_imu/training/` | Training, objectives and checkpoint lifecycle |
| `src/wearable_imu/evaluation/` | Offline, subject/patient and sensor-domain evaluation |
| `src/wearable_imu/inference/` | Export loading, input validation and prediction interface |
| `configs/` | Independent dataset, preprocessing, model, training, evaluation and inference configs |
| `scripts/` | Prepare/train/evaluate/export entry-point stubs |
| `experiments/` | Versioned experiment configs and exploratory notebooks |
| `tests/` | Module unit/integration tests and synthetic fixtures |
| `docs/` | Dataset, pipeline and hardware-domain documentation |
| `environments/` | Research/inference dependency locks once implementations exist |

Dataset root: `dataset_roots.sisfall` in repository `configs/paths.local.json`. Local working data: `data/imu/`. Own checkpoints, exports, reports and runs: `artifacts/imu/`. These payloads are ignored by Git.

## Non-negotiable data rules

SisFall is sampled at 200 Hz. Preserve subject and trial/activity identity. Explicitly select one accelerometer and one gyroscope stream for the six-axis wearable experiment; do not assume all raw columns match the MPU-9250 configuration. Document raw units and conversion factors.

Check subject IDs, activity mapping, channel order, units, saturation, sample count/rate and subject-disjoint splits. Do not report exact onset accuracy without appropriate additional time annotations.

Keep split construction/auditing module-owned under `datasets/splits/`. Keep hardware-domain evaluation under `evaluation/hardware_domain/`. Shared helpers may check group overlap, but must not replace this module's split policy.

## Implementation status

Configs are templates. Null fields require explicit decisions and validation before use. Scripts and `IMUPredictor` intentionally raise an implementation-pending error. No training, inference, sensor acquisition or score calculation is implemented. Do not import training code from the inference package.
