# ECG checks

Run from the repository root after installing the ECG/common packages:

```powershell
.\.venv\Scripts\python.exe -m pytest modules/ecg/tests -q
```

Unit checks cover exact statement cohorts, unknown-confidence positives, contradictions, patient-fold independence, source/output separation, physical-voltage resampling, neural forward/backward paths, parameter limits, FFT/direct state-space equivalence, export-kernel equivalence, calibrated-input/gap rejection, tied-score thresholds and monotonic probability calibration.

The experiment adds data-dependent verification: official checksums, source rehashing after training, duplicate waveform checks, cohort counts and ONNX parity on eligible training records. Public datasets and individual predictions remain outside versioned test fixtures.

The tests do not establish clinical fidelity, external-domain accuracy or Jetson timing.
