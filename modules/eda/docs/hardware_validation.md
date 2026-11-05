# EDA/GSR hardware-domain validation

The GSR/EDA sensor module is not finalized. Record excitation, ADC, electrode placement, conversion/calibration, units, bandwidth and sampling clock once selected. Validate differences from Empatica E4, including whether optional motion data are actually available at deployment.

Preserve clock, units, calibration and signal-quality information in replayable local captures. Store data in `data/eda/wearable/` and generated reports in `artifacts/eda/reports/`. Keep validation subjects separate from development subjects.

Measure actual acquisition-to-output latency and evaluate the real deployment preprocessing, not only an offline approximation. An unusable recording must not become a negative detection.
