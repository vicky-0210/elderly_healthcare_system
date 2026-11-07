# Local setup

## Dataset paths

Copy `configs/paths.example.json` to `configs/paths.local.json` and fill the dataset roots on your own computer.

PowerShell:
```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
```

POSIX:
```sh
cp configs/paths.example.json configs/paths.local.json
```

Windows JSON paths must escape backslashes or use forward slashes, for example `C:/Datasets/PTB-XL/1.0.3`. Do not commit the local copy. No dataset is downloaded by the scaffold.

## Independent environments

Use an isolated development environment for the area you own. Python packages use a src layout. Prefer editable installs; the ECG command-line entry points also support direct execution from the repository.

For example, after creating your development environment:
```sh
python -m pip install -e packages/common -e modules/ecg
```

Each module documents its research environment and data preparation. The PPG workflow reads the local OSASUD release through an explicit source argument. [ECG setup](../modules/ecg/README.md) · [PPG setup and verification](../modules/ppg/README.md) · [IMU setup](../modules/imu/README.md) · [EDA setup](../modules/eda/README.md)

Add dependencies to the owning package and record reproducible environment locks when actual code is introduced. The directories under `environments/` are reserved for that work. Avoid a single environment containing every team's experiment stack.

The frontend has no selected framework or package manager yet. The API has no selected web framework yet.

## Jetson

Record the actual board revision and installed JetPack, Python, CUDA and TensorRT versions before adding edge dependencies. Build/export and target runtime environments may differ; validate compatibility rather than assuming workstation wheels will install on the original Nano.
