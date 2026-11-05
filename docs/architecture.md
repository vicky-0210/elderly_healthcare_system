# Architecture

## Package boundaries

| Path | Owns | Must not own |
| --- | --- | --- |
| `apps/frontend` | Views, presentation, API client | Signal preprocessing or diagnostic logic |
| `apps/backend` | Routes, services, schema validation, persistence | Dataset training scripts |
| `acquisition` | Device drivers, transport, clock handling, buffering, replay | Public-dataset labels or neural networks |
| `modules/<signal>` | Complete independent offline pipeline and matching inference implementation | Another signal's preprocessing |
| `packages/common` | Configuration/IO/logging helpers and prediction interface | Physiology-specific filters, labels or models |
| `contracts` | Versioned interchange formats | ML architecture decisions |
| `edge` | Predictor lifecycle, per-signal scheduling, adapters, health and telemetry | Model training or fabricated missing signals |
| `deploy` | Environment locks, packaging, services and release operations | Experiment notebooks |

Common helpers should be introduced only when their behavior truly matches across modules. Split auditing can be shared, while the actual PTB-XL official folds and other datasets' group-split policies remain module-owned.

## Offline flow

Public dataset -> module datasets/validation/labels -> patient/subject splits -> preprocessing and windowing -> models/training -> evaluation -> artifact export.

Experiment configuration and notebooks live in the module; generated arrays and run results live under local data/artifact roots. All fitted preprocessing parameters belong to the training split and exported artifact.

## Real-time flow

Wearable -> acquisition transport/driver -> timestamped sample blocks -> signal-specific edge buffers -> module inference -> prediction contract -> backend/frontend.

The four pipelines may have different window sizes, clocks, update rates and runtimes. Do not align them by equal array index or force synchronous inference. Edge scheduling must tolerate one unavailable signal without converting its missing result into normal.

A future fusion/integration service can consume independently timestamped predictions. Multimodal fusion is not implemented or assumed by this scaffold.

## Dependency direction

Modules may depend on common interfaces, not each other. The edge selects module predictors behind the shared interface. Runtime imports must remain free of dataset loaders, experiment tools and training-only dependencies. A live model's preprocessing must match the fitted/exported offline pipeline, including causal versus buffered behavior.

## Existing work

The prior ECG report and audit scripts remain untouched in `research/`. Their downloaded metadata and generated audit outputs are ignored. New ECG implementation belongs in `modules/ecg/`.
