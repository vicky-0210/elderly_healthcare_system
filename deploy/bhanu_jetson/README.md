# Sleep apnea / hypopnea detector - Jetson Nano package

Ensemble of an SpO2 CNN and an SpO2 gradient-boosting model. One 60 s window every 10 s, decision delayed 5 s.

Orin Nano quick start: copy this folder to the Jetson, `bash setup_orin.sh`, test with `python3 replay_demo.py sample_spo2.csv`, then
`python3 run_live.py --bus 7` (optional: apnea-detector.service to start on boot).
Files: run_live.py, max30102.py (I2C driver), replay_demo.py, setup_orin.sh, apnea-detector.service, requirements.txt, runtime.py (Detector), spo2_features.py (feature code), gbm_numpy.py (numpy tree inference), models.py (CNN),
cnn_spo2.pt (CNN weights), gbm.npz (trees), calibration.npz (score grids + one global alarm threshold), live_template.py.

Needs: Python 3.8+ (numpy, pandas, scipy, torch, smbus2). No scikit-learn.

Input per second: SpO2 (%) and whether that second of PPG is usable (ppg_second_ok: no NaN / clipping / flat line).
Output every 10 s: p_cnn, p_gbm, ensemble score (0-1), alarm (score >= threshold), ok flag (False = >10% unusable seconds).

Caveats: trained on hospital-device SpO2 (OSASUD); MAX30102 SpO2 differs, so re-check on your sensor. The score is a relative
ranking, the alarm threshold flags about 34% of windows by design. Not a medical device. Speed was measured on a laptop only.
