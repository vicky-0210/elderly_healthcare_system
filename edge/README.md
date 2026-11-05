# Edge runtime

Owns model loading, independent signal scheduling, health and telemetry on the Jetson.

`src/wearable_edge/`:
- `runtime/`: orchestration, queues, per-signal update schedules and lifecycle.
- `adapters/`: module predictor adapters and validated exported-model runtimes.
- `health/`: model/device availability, errors and resource health.
- `telemetry/`: structured output delivery and local buffering.

The four module inference interfaces implement a shared contract but retain their own preprocessing, window lengths and label semantics. Live deployment must not depend on training packages or public dataset loaders.

Copy `configs/runtime.example.json` to a local config when implementing integration. No models are enabled and no runtime is implemented yet. See repository `deploy/jetson/` for hardware environment work.
