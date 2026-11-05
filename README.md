# elderly_healthcare_system

Repository scaffold for four independent signal pipelines and their eventual integration on NVIDIA Jetson Nano.

**Status:** folder structure, configuration templates, documentation and interface stubs only. No sensor driver, model, training pipeline, API or frontend is implemented or clinically validated. Scripts and predictors deliberately fail until implemented.

| Module | Dataset | Intended task | Wearable source |
| --- | --- | --- | --- |
| ECG | PTB-XL | LVH-associated ECG patterns | BioAmp EXG Pill, verified single-lead configuration |
| PPG | OSASUD | Apnea and hypopnea events | MAX30102 |
| IMU | SisFall | Fall versus activities of daily living (ADL) | MPU-9250, six selected axes |
| EDA | WESAD wrist EDA | Laboratory baseline versus stress | EDA/GSR module to be selected |

## Layout

```text
apps/
  frontend/                 User interface; no signal processing
  backend/                  API, services and persistence
acquisition/                Drivers, transports, timestamps, buffers and replay
modules/
  ecg/                      Independent PTB-XL pipeline
  ppg/                      Independent OSASUD pipeline
  imu/                      Independent SisFall pipeline
  eda/                      Independent WESAD pipeline
packages/common/            Shared interfaces and generic utilities
contracts/                  Shared data/inference contract and future schemas
configs/                    Local paths and model registry templates
data/{ecg,ppg,imu,eda}/      Optional local data workspace; ignored payloads
artifacts/{ecg,ppg,imu,eda}/ Checkpoints, exports, runs and reports; ignored payloads
edge/                       Real-time scheduling and inference integration
deploy/                     Jetson, API, frontend and service deployment assets
docs/                       Architecture, workflows and validation guidance
tests/                      Cross-module, contract, end-to-end and hardware tests
scripts/                    Repository-level operations
research/                   Existing ECG research, preserved in place
```

Each signal module owns its dataset loading, validation, labels, splits, windowing, preprocessing, model definitions, training, evaluation, inference, configurations, experiments, tests and documentation. Matching directory names do **not** require matching algorithms, window sizes, sample rates, labels or frameworks.

## Local datasets

Every teammate keeps their own datasets. There is no shared absolute path and no dataset download during installation.

1. Copy `configs/paths.example.json` to `configs/paths.local.json`.
2. Set the four dataset roots to existing absolute directories on your machine. Unused roots may remain null.
3. Use paths outside the repository, or the ignored `data/<signal>/raw/` directories. Do not copy data into source folders.
4. Keep patient/subject manifests, wearable recordings and generated arrays local.
5. Review staged files before committing. Ignore rules do not untrack files already committed.

These JSON files are templates; automatic configuration loading is not yet implemented. Future loaders must resolve relative output roots against the repository root and fail clearly for missing dataset roots.

## Modes

**Offline:** local public dataset -> module validation/labels/splits -> module preprocessing/windowing -> training -> evaluation -> versioned exported model.

**Real-time:** acquisition -> timestamped per-signal buffers -> matching module inference preprocessing -> predictor -> detection result -> API/frontend or local storage. An offline replay mode can exercise the real-time path without physical sensors.

Preprocessing may be reused within a signal module, but offline dataset parsers must never become live sensor drivers. Edge and API imports must not pull in training-only dependencies.

## Getting started

Read [architecture](docs/architecture.md), [local setup](docs/local_setup.md) and the README for your module. Editable Python package shells are provided for common code, the four modules, acquisition, the API and edge runtime. Frontend framework selection is intentionally left to the frontend team.

The original Jetson Nano runtime must have its own verified Python/CUDA/TensorRT dependencies; do not install a workstation training environment onto it unchanged.

See [CONTRIBUTING.md](CONTRIBUTING.md) for ownership boundaries and [validation](docs/validation.md) for the checks to implement. Existing ECG work remains in [research](research/research_review_and_protocol.md).
