# Waist IMU fall detection

This module scores fall versus ADL trials from SisFall and runs sliding-window inference on six calibrated MPU-9250 axes. Start with the [research report and graphs](../../research/results/sisfall_v1/REPORT.md), [literature review](../../research/sisfall_literature.md) and [study protocol](../../research/sisfall_protocol.md).

## Verify the completed study

Run from the **team repository root** in PowerShell with its research virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pytest -q modules/imu/tests
.\.venv\Scripts\python.exe modules/imu/scripts/verify_results.py --source "C:\Final Project\Datasets\sisfall\SisFall_dataset"
.\.venv\Scripts\python.exe modules/imu/scripts/bench_edge.py artifacts/imu/exports/sisfall_v1/native_100
```

The verifier recomputes confusion counts from individual held-out decisions, checks group identities, compares file hashes, and replays complete and fragmented recordings through the exports. Only `--save` writes a verification summary. The source argument is optional for collaborators without a local SisFall copy.

## Reproduce the experiments

```powershell
.\.venv\Scripts\python.exe modules/imu/scripts/prepare_data.py --source "C:\Final Project\Datasets\sisfall\SisFall_dataset"
.\.venv\Scripts\python.exe modules/imu/scripts/train.py
.\.venv\Scripts\python.exe modules/imu/scripts/audit_filters.py
.\.venv\Scripts\python.exe modules/imu/scripts/evaluate.py
.\.venv\Scripts\python.exe modules/imu/scripts/export.py
.\.venv\Scripts\python.exe modules/imu/scripts/export.py --experiments dm_standard inception
.\.venv\Scripts\python.exe modules/imu/scripts/validate_transfer.py
.\.venv\Scripts\python.exe modules/imu/scripts/validate_transfer.py --experiments dm_standard inception
foreach ($seed in 4907,6203,7919) {
    .\.venv\Scripts\python.exe modules/imu/scripts/train.py --seed $seed --experiments dm_tcn dm_standard inception dm_distilled
}
.\.venv\Scripts\python.exe modules/imu/scripts/summarize_seeds.py
.\.venv\Scripts\python.exe modules/imu/scripts/export.py --experiments dm_distilled
.\.venv\Scripts\python.exe modules/imu/scripts/inspect_signals.py
.\.venv\Scripts\python.exe modules/imu/scripts/validate_transfer.py --experiments dm_distilled
foreach ($name in 'dm_tcn','dm_standard','inception','dm_distilled') {
    .\.venv\Scripts\python.exe modules/imu/scripts/check_precision.py --model $name
}
foreach ($name in 'native_100','dm_tcn','dm_standard','inception','dm_distilled') {
    .\.venv\Scripts\python.exe modules/imu/scripts/bench_edge.py "artifacts/imu/exports/sisfall_v1/$name" --build-reference
}
```

Use `train.py --list` to inspect model IDs and `--experiments cnn gru dm_tcn` to run a subset. For a fresh replication use a separate `--root` and pass the same path to evaluation and export. Training resumes completed folds. The executable definitions in `src/wearable_imu/training/catalogue.py` specify the exact settings. Each fold stores its effective configuration, scalers, participant lists and threshold. See the [dataset contract](docs/dataset.md), [signal pipeline](docs/pipeline.md) and [Jetson/MPU guide](docs/hardware_validation.md).

| Location from the repository root | Contents |
|---|---|
| `research/results/sisfall_v1/` | Report, plots, comparison and validation tables |
| `artifacts/imu/runs/sisfall_v1/<experiment>/fold0…fold4/` | Fitted checkpoints, scores per recording and window, validation decisions and provenance |
| `artifacts/imu/runs/sisfall_v1_seed<seed>/` | Repeat runs |
| `artifacts/imu/exports/sisfall_v1/<experiment>/` | Inference bundle, fitted preprocessing metadata, threshold and reference check |
| `data/imu/processed/sisfall_v1/` | Derived six-axis cache and source-hashed trial manifest |

Research dependencies and the target runtime are documented in [environments](environments/README.md). Live acquisition is under `acquisition/src/wearable_acquisition/drivers/mpu9250`.
