# Independent signal modules

Each module owns its ML lifecycle. See [ECG](ecg/README.md), [PPG](ppg/README.md), [IMU](imu/README.md) and [EDA](eda/README.md).

Shared code is limited to truly reusable infrastructure and the external prediction contract. Each module has separate configuration, experiments, dependencies, tests and local data/artifact namespaces. No common model or preprocessing algorithm is imposed.
