# IMU signal and decision pipeline

The 100 Hz edge candidate runs six physical axes through the causal IPMV front end, 20 Hz tokens, two-second windows with a one-second hop, a compact XGBoost estimator and an alarm policy. For 100 Hz research replay, apply causal 40 Hz antialiasing at the original 200 Hz and decimate. A device configured at 100 Hz supplies actual 100 Hz physical samples.

The default representation uses magnitude extrema and derivatives from native-rate samples, adaptive per-sensor filtering, a gravity estimate and gravity-relative projections, and native-rate causal db2 wavelet coefficient energies. The RNN and convolutional models operate on completed windows. Trial supervision takes the maximum window score. The final scalar sigmoid is calibrated on recording maxima. Streaming window evidence drives local alerts and has no independently validated event probability.

The report documents window geometry, filter alternatives, fold-fitted normalization, neural branch fusion, weak witness learning and comparisons. [Study protocol](../../../research/sisfall_protocol.md) · [Measured results](../../../research/results/sisfall_v1/REPORT.md)
