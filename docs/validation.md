# Validation plan

This scaffold reserves test locations; it does not supply passing ML or hardware tests.

- Module unit tests: loader parsing, units, labels, gap detection, resampling, preprocessing, windows and predictor input validation.
- Module integration tests: small synthetic data -> pipeline -> exported/inference parity. Public datasets are optional local integration inputs, not required downloads in CI.
- Root contract tests: signal windows, prediction semantics and invalid/unavailable states.
- Root integration tests: acquisition/replay -> buffers -> independent predictor adapters -> API.
- Root end-to-end tests: a synthetic complete system flow.
- Root hardware tests: real sensors, sampling clocks, calibrated units and Nano timing; opt-in only.

## Required data checks

**ECG:** retain official patient folds; use the LVH statement rather than HYP superclass; distinguish single-lead student from 12-lead research; verify calibrated amplitude and sensor-domain behavior.

**PPG:** preserve patient IDs and timestamps; inspect discontinuities; align event labels by time; evaluate patient-independent detection and MAX30102 domain shift.

**IMU:** use subject-independent splits; explicitly select and calibrate six axes; evaluate false alarms per monitored time. Trial-level SisFall labels are not exact fall onset labels.

**EDA:** use wrist EDA at 4 Hz; align labels to the EDA time base; retain baseline/stress only for the binary task; test tonic/phasic and optional motion processing without subject leakage.

Model accuracy, latency, event thresholds and clinical utility must be measured. Do not populate reports with placeholder success metrics.
