# PPG respiratory-event research

The OSASUD study uses a single 80 Hz Mindray plethysmogram to estimate physician-labeled apnea or hypopnea at one-second resolution. Four neural architectures and seven controls have completed five patient-independent folds. The selected CNN + Transformer produced 68.8 false events per scored hour at its research threshold. The exported model is suitable for replay and sensor-domain investigation. [Report and figures](../../research/results/osasud_v1/REPORT.md) · [Study protocol](../../research/osasud_protocol.md)

## Binary two-time-scale follow-up

Four binary variants completed 20 additional patient-independent fits. The 12 s pulse / 90 s modulation model with hard labels improved mean patient AP to .247 with 56.2 false events per scored hour and 46.3% event recall. A separate experiment compared trailing event-fraction supervision, an auxiliary target and matched alert budgets. The selected binary research bundle is compatible with the streaming CLI and emits one event probability per scored second. [Follow-up results, graphs and commands](../../research/results/osasud_binary_dual_rate_v1/REPORT.md)

## Verify the completed run

Open PowerShell at the repository root. The first command checks fold membership, recomputes each training scaler, aligns every stored label and timestamp to its patient packet, checks calibrated scores, replays three batches per split per checkpoint and recomputes patient AP. Passing the original pickle also checks its SHA-256. The second command checks the optical frontend, stream timing, inference contract and sensor FIFO behavior.

~~~powershell
.\.venv\Scripts\python.exe modules/ppg/scripts/verify_results.py --source 'C:\Final Project\Datasets\osasud\dataset_OSAS.pickle'
.\.venv\Scripts\python.exe -m pytest modules/ppg/tests -q
~~~

## Reproduce on another workstation

Use the repository root as the working directory for every command. Install the matching PyTorch build for your workstation and the remaining research dependencies. The checked-in JSON files under `configs/` describe the executed study. The versioned Python modules and CLI arguments drive execution. [Tested workstation versions](environments/README.md)

~~~powershell
python -m pip install -e packages/common -e "modules/ppg[research]"
python modules/ppg/scripts/prepare_data.py --source 'C:\Final Project\Datasets\osasud\dataset_OSAS.pickle'
python modules/ppg/scripts/prepare_views.py
python modules/ppg/scripts/train.py
python modules/ppg/scripts/evaluate.py
python modules/ppg/scripts/alert_burden.py
python modules/ppg/scripts/transfer.py
python modules/ppg/scripts/export.py
~~~

Preparation opens the source archive read-only and writes derived files to `data/ppg/processed/osasud_v1/`. Training resumes completed folds using their completion markers. Use a fresh run directory for a changed protocol. To run a subset, pass model names to `train.py --models`. The full comparison expects all eleven configurations.

| Output | Location from repository root |
|---|---|
| Comparison, class and event metrics, selection, figures | `research/results/osasud_v1/` |
| Read-only source audit and patient fold assignment | `research/results/osasud_v1/source_audit.json` and local `folds.csv` |
| Model, scaler, training history, calibrated validation and held-out predictions for each fold | `artifacts/ppg/runs/osasud_v1/<model>/fold0` through `fold4` |
| All-patient research export, matching scaler, calibration and threshold | `artifacts/ppg/exports/osasud_v1/cnn_transformer/` |

The streaming path accepts timestamped optical samples. On Linux the MAX30102 driver reads red and IR. The predictor uses one chosen channel, defaults to IR, and outputs five timestamped scores at each five-second boundary. [Hardware instructions and validation requirements](docs/hardware_validation.md)
