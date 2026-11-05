# PPG hardware-domain validation

Validate MAX30102 channel choice, placement, sample rate, gain/LED settings, saturation and motion artifacts. The clinical Mindray 80 Hz stream is not assumed interchangeable with the MAX30102. Test actual wearable recordings in a separate domain-validation study.

Preserve clock, units, calibration and signal-quality information in replayable local captures. Store data in `data/ppg/wearable/` and generated reports in `artifacts/ppg/reports/`. Keep validation subjects separate from development subjects.

Measure actual acquisition-to-output latency and evaluate the real deployment preprocessing, not only an offline approximation. An unusable recording must not become a negative detection.
