# MPU-9250 and Jetson Nano deployment

Mount the six-axis unit at the waist, record axis mapping, belt tightness and side, and preserve the mounting setup across calibration and recording. Select the matching bundle rate, 100 Hz for the compact XGBoost or 200 Hz for the neural comparators. The I²C driver requires WHO_AM_I 0x71, checks configuration readback, selects ±16 g and ±2000 degrees/s, and reads 12-byte FIFO frames. With the 1 kHz base clock, the sample divider is 9 or 4. The digital filter setting 3 sets roughly 41 Hz gyro bandwidth and 44.8 Hz accelerometer bandwidth on the revision documented here. FIFO timestamps use a monotonic anchor and nominal sample count. Overflow/short read resets both FIFO and processing state. The gold reference tests signed byte ordering and continuity with a mock bus. [MPU-9250 register map](https://invensense.tdk.com/wp-content/uploads/2017/11/RM-MPU-9250A-00-v1.6.pdf)

| Characteristic | SisFall release | Target setup |
|---|---|---|
| Acceleration | ADXL345 full-resolution ±16 g, approximately 3.9 mg/count | MPU-9250 ±16 g, 2048 counts/g, 16-bit output |
| Angular velocity | ITG3200 ±2000 degrees/s. Release equation implies 16.384 counts/(degree/s), vendor nominal is 14.375 | MPU-9250 ±2000 degrees/s, nominal 16.4 counts/(degree/s) |
| Bias, gain and alignment | Session-level device calibration and exact transfer function are unavailable | Fit six-face accelerometer affine and stationary gyro bias per unit. Calibrated gyro scale requires a known angular rate or reference instrument. |
| Bandwidth, clock and movement | Nominal 200 Hz waist unit, filters and belt influence the waveform | Check FIFO service, drops, jitter, physical bandwidth, clipping, axis alignment and reproducibility at the intended location. |

Measure stationary noise per axis, Allan deviation or drift across realistic warm-up/temperature, six-face residuals, and rotation gains in both directions. Compare synchronized paired recordings at the waist to examine scale, phase delay, noise spectra and transient attenuation. A static coordinate rotation differs from moving the sensor to another body site. [ADXL345 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adxl345.pdf) · [ITG3200 datasheet](https://product.tdk.com/en/system/files/dam/doc/product/sensor/mortion-inertial/gyro/data_sheet/itg-3200-datasheet.pdf) · [MPU-9250 specification](https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/ps-mpu-9250a-01-v1.1.pdf)

## On the target

Copy the selected export directory with its `runtime.json`, model payload and synthetic reference, along with `modules/imu`, `acquisition` and `packages/common`. The tree model uses NumPy and SciPy. The neural path supports ONNX Runtime CPU or a locally built TensorRT 8 engine with PyCUDA. Match the installed JetPack and Python packages on the actual Nano. For neural TensorRT, build on the target and supply the full engine path.

```bash
XGB=artifacts/imu/exports/sisfall_v1/native_100
export PYTHONPATH="$PWD/packages/common/src${PYTHONPATH:+:$PYTHONPATH}"
NN=artifacts/imu/exports/sisfall_v1/dm_distilled
python3 modules/imu/scripts/bench_edge.py "$XGB" --iterations 1000 --output nano_xgb.json
/usr/src/tensorrt/bin/trtexec --onnx="$NN/model.onnx" --fp16 --saveEngine="$NN/fall.engine"
python3 modules/imu/scripts/bench_edge.py "$NN" --trt-engine "$NN/fall.engine" --iterations 1000 --output nano_neural.json
python3 modules/imu/scripts/wearable.py capture-calibration device_poses.npz --bus 1 --fs 100
python3 modules/imu/scripts/wearable.py calibrate device_poses.npz device_calibration.json
python3 modules/imu/scripts/wearable.py live "$XGB" device_calibration.json --bus 1 --address 0x68
```

The calibration input holds `faces_g` with shape (6, samples, 3) in +X, −X, +Y, −Y, +Z, −Z order and `stationary_dps` with shape (samples, 3). Supply at least 100 stationary samples for each pose and the gyro. The driver emits local JSON lines after window warm-up. For use with a neural engine add `--trt-engine` to `live`. The magnetometer has no matching training stream in SisFall.

Measure actual acquisition-to-alert latency with an independent synchronized reference, plus power, thermals, memory, missed deadlines and FIFO overflows during sustained acquisition. Measure freely conducted ADL false alerts per participant-hour and clinically relevant subgroups. Independent, consented, annotated episodes are required for onset accuracy or real elderly-fall sensitivity. The shipped benchmark measures software time per hop and compares outputs against a golden sequence.
