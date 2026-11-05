# IMU hardware-domain validation

Verify MPU-9250 axis orientation, sensor placement, accelerometer/gyro ranges, units, sampling clock and saturation. The initial task uses six axes; magnetometer readings are not an implicit input. Measure false alarms during continuous ADL and test acquisition resets and gaps.

Preserve clock, units, calibration and signal-quality information in replayable local captures. Store data in `data/imu/wearable/` and generated reports in `artifacts/imu/reports/`. Keep validation subjects separate from development subjects.

Measure actual acquisition-to-output latency and evaluate the real deployment preprocessing, not only an offline approximation. An unusable recording must not become a negative detection.
