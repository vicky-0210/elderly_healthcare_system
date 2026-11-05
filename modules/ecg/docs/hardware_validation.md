# ECG hardware-domain validation

Verify BioAmp revision, actual ADC, gain, analog passband, polarity, lead placement and measured sampling clock. The intended Lead II-like placement and approximately 360 Hz rate are hypotheses until measured. Compare the public clinical signal and actual wearable captures; do not infer missing spatial leads.

Preserve clock, units, calibration and signal-quality information in replayable local captures. Store data in `data/ecg/wearable/` and generated reports in `artifacts/ecg/reports/`. Keep validation subjects separate from development subjects.

Measure actual acquisition-to-output latency and evaluate the real deployment preprocessing, not only an offline approximation. An unusable recording must not become a negative detection.
