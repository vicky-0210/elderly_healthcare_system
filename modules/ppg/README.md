# PPG module

Dataset: **OSASUD**. Intended task: **Apnea and hypopnea detection**.

This module is independent of the other signal pipelines. Algorithms, framework, architecture and windowing are not prescribed by the shared structure.

## Locations

| Path | Responsibility |
| --- | --- |
| `src/wearable_ppg/datasets/` | Public-dataset loaders and provenance |
| `src/wearable_ppg/validation/` | Signal/data integrity checks |
| `src/wearable_ppg/labels/` | Dataset-specific label mapping and exclusions |
| `src/wearable_ppg/windowing/` | Segmentation, timestamps and boundary rules |
| `src/wearable_ppg/preprocessing/` | Signal-specific fitted and runtime transformations |
| `src/wearable_ppg/models/` | Model definitions |
| `src/wearable_ppg/training/` | Training, objectives and checkpoint lifecycle |
| `src/wearable_ppg/evaluation/` | Offline, subject/patient and sensor-domain evaluation |
| `src/wearable_ppg/inference/` | Export loading, input validation and prediction interface |
| `configs/` | Independent dataset, preprocessing, model, training, evaluation and inference configs |
| `scripts/` | Prepare/train/evaluate/export entry-point stubs |
| `experiments/` | Versioned experiment configs and exploratory notebooks |
| `tests/` | Module unit/integration tests and synthetic fixtures |
| `docs/` | Dataset, pipeline and hardware-domain documentation |
| `environments/` | Research/inference dependency locks once implementations exist |

Dataset root: `dataset_roots.osasud` in repository `configs/paths.local.json`. Local working data: `data/ppg/`. Own checkpoints, exports, reports and runs: `artifacts/ppg/`. These payloads are ignored by Git.

## Non-negotiable data rules

The distributed Mindray PPG stream is 80 Hz. Preserve patient identity and timestamps. Keep all five source label names, including the apnea subtypes. Select binary, three-class or five-class processing explicitly instead of silently merging events. Inspect source fields and annotation timing rather than assuming a file layout.

Check timestamps, monotonic segments, missing samples, units, optical channels, event intervals, label vocabulary and patient independence. Document any excluded gaps or interrupted recordings.

Keep split construction/auditing module-owned under `datasets/splits/`. Keep hardware-domain evaluation under `evaluation/hardware_domain/`. Shared helpers may check group overlap, but must not replace this module's split policy.

## Implementation status

Configs are templates. Null fields require explicit decisions and validation before use. Scripts and `PPGPredictor` intentionally raise an implementation-pending error. No training, inference, sensor acquisition or score calculation is implemented. Do not import training code from the inference package.
