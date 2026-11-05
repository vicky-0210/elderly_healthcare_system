# Integration contracts

This is the canonical location for future versioned JSON/API schemas. Generic Python interfaces live in `packages/common`; each signal owns its inference implementation. The contract standardizes transport, not preprocessing or architectures.

## Signal window (version 1 draft)

A window carries `schema_version`, `signal`, `stream_id`, `clock_domain`, sample timestamps in seconds, time-major samples (N x C), ordered channel names, per-channel units, nominal sample rate, quality/missingness masks and optional pseudonymous subject ID.

Timestamp count must equal sample count; channel names/units must agree with C. Preserve irregular timing and gaps rather than silently claiming uniform sampling. State whether time is UTC epoch, dataset-relative or a device's monotonic clock; never combine clocks without an explicit synchronization mapping. Units and calibrated voltage must not be inferred from array magnitude.

## Prediction (version 1 draft)

A result carries `schema_version`, `signal`, `stream_id`, window start/end and clock domain, `model_version`, `preprocessing_version`, named class scores, threshold/calibration version, quality flags and status.

Statuses should distinguish `ok`, `insufficient_signal`, `model_unavailable` and `error`. Unavailable or unusable input must not be encoded as a negative detection. Scores are not necessarily calibrated probabilities; record score semantics explicitly.

ECG outputs refer to LVH-associated ECG patterns. EDA outputs refer to laboratory baseline/stress labels. PPG outputs retain the distinction between event-level and window-level results. IMU output timing must not imply annotated onset accuracy unsupported by the data.

## Boundaries

- Acquisition produces sample blocks; edge buffering/windowing constructs predictor inputs.
- Each module predictor validates its own channels, units, sample rate and preprocessing compatibility.
- API/frontend consume prediction results and device health; they do not retrain or normalize signals.
- No shared target label, fixed window duration or common neural network is imposed.

`schemas/` and `examples/` are reserved for implementation. Examples must be synthetic.
