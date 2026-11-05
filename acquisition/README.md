# Sensor acquisition

Owns real sensor communication independently of public dataset processing.

`src/wearable_acquisition/`:
- `drivers/bioamp/`, `drivers/max30102/`, `drivers/mpu9250/`, `drivers/eda/`: device-specific acquisition/calibration.
- `transports/`: actual serial/I2C/other transport adapters once verified.
- `timebase/`: timestamping, sampling-clock measurement and synchronization.
- `buffering/`: sample blocks, gaps, overflow and reset handling.
- `replay/`: replay of local recorded/synthetic streams for integration tests.

BioAmp/GSR analog output needs a verified ADC/acquisition interface; do not treat the Nano as an unspecified ADC. The precise EDA hardware is unresolved. Six IMU axes must be explicitly selected; the magnetometer is not required for the initial task.

Copy device config to a local file before filling in actual connections. All devices are disabled in the example. No hardware access occurs on import, and no driver is implemented.
