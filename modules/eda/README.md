# Wrist EDA and multimodal stress-associated arousal

The [results report](../../research/results/wesad_loso_v1/README.md) compares the trained models, identifies the deployment choice and links its limitations. The [research protocol](../../research/wesad_research_protocol.md) covers prior work, the novelty hypothesis and the model-selection rule. All paths below start at the existing repository root.

## End-to-end flow

1. Read the synchronized WESAD distribution. Verify identities, sample counts and sensor headers, retain original 700 Hz annotation votes, and extract wrist EDA at 4 Hz, accelerometer at 32 Hz, temperature at 4 Hz and BVP at 64 Hz. Raw public files stay in the teammate's local dataset directory.
2. Assess EDA contact, gaps, derivative outliers and motion. Filter and decompose each participant's continuous trace before windowing. Align native-rate summaries to completed 4 Hz bins. The [preprocessing methods](docs/preprocessing.md) describe MC-SAED, Choi, cvxEDA and the sparse comparator.
3. Reserve 120 s of original protocol baseline for the explicitly calibrated experiment. After that interval, create 60 s windows every 15 s independently per person. Require at least 80% label purity against all original annotation ticks. The binary target contains baseline and stress. Secondary experiments add amusement and meditation.
4. For each of the 15 held-out people, use a grouped two-person inner validation set, fit scalers within training subjects, choose settings and temperature, refit on the other 14, then save probabilities on the held-out recording. Compare classical, standalone neural, hybrid and self-supervised candidates. Each fold has its own checkpoint.
5. Evaluate equal-weight participant metrics, paired component ablations, uncertainty and calibration. Select the zero-calibration export by the documented inner-score and resource rule. Use raw native-rate replay to compare its predictions against the offline transformation.

## Reproduce on Windows

Use Python 3.12. The measured workstation used PyTorch 2.6.0 with CUDA 12.4. A CPU installation can run the research pipeline. The dependency file pins the versions used for the analysis. The deployment runtime imports the framework for the selected family when loading its model.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -e packages/common -e modules/eda
& $py -m pip install -r modules/eda/environments/research.txt
& $py modules/eda/scripts/prepare_data.py --source 'C:\Final Project\Datasets\wesad'
& $py modules/eda/scripts/audit_wesad.py --source 'C:\Final Project\Datasets\wesad' --publish research/results/wesad_loso_v1/preprocessing
& $py modules/eda/scripts/train.py classical --jobs 4
& $py modules/eda/scripts/train.py neural
& $py modules/eda/scripts/verify_classical.py
& $py modules/eda/scripts/audit_neural.py
& $py modules/eda/scripts/benchmark_classical.py
& $py modules/eda/scripts/evaluate.py
& $py modules/eda/scripts/protocol_challenge.py
& $py modules/eda/scripts/export.py
& $py modules/eda/scripts/replay.py --report research/results/wesad_loso_v1/replay.json
& $py modules/eda/scripts/export.py --choice neural
& $py modules/eda/scripts/replay.py --export artifacts/eda/exports/wesad_loso_v1_neural --report research/results/wesad_loso_v1/neural_replay.json
& $py modules/eda/scripts/export.py --choice neural --calibration rest
& $py modules/eda/scripts/replay.py --export artifacts/eda/exports/wesad_loso_v1_neural_rest --report research/results/wesad_loso_v1/neural_rest_replay.json
```

Pass `--experiment logistic__eda__zero` to the classical trainer or `--models cnn_zero` to the neural trainer for a single comparison. The full evaluation checks the experiment catalogue and requires complete folds. With `--partial --out <exploratory-directory>` it can inspect an interrupted study. Train and test windows always belong to distinct participants.

## Find and verify outputs

| Location | What to inspect |
| --- | --- |
| `research/results/wesad_loso_v1/` | Portable report, all-model table, per-participant metrics, paired contrasts, nested choices and plots |
| `research/results/wesad_loso_v1/preprocessing/` | Source audit, method diagnostics, participant-level data quality and illustrative signal traces |
| `data/eda/processed/wesad_v1/` | Local source hashes, wrist caches and derived representations |
| `artifacts/eda/runs/wesad_loso_v1/{classical,neural}/` | One directory per experiment, with meta.json, folds.csv, predictions.csv and 15 local checkpoints |
| `artifacts/eda/exports/wesad_loso_v1/` | Lightweight logistic export. The coefficient JSON contains feature order, robust scaling, sigmoid weights and temperature. Scoring it requires NumPy and SciPy. |
| `artifacts/eda/exports/wesad_loso_v1_neural/` and `wesad_loso_v1_neural_rest/` | Deep-learning exports for zero and explicitly resting-calibrated inference, each with its own contract and weights |

Run `scripts/evaluate.py` from the command above to regenerate metrics from per-window probabilities. It checks all 15 outer identities, the inner/refit subject lists, matching primary window identities, probability normalization and class mappings. The first 12 metric columns in the model table use equal participant weights. Raw prediction files use class index 0 for WESAD baseline and 1 for WESAD stress. Model archives contain fitted objects and should be loaded only from a trusted source.

The [classical run instructions](docs/classical_loso.md) and [neural run instructions](docs/neural.md) explain fold-by-fold re-scoring. The replay command checks preprocessing and the all-data export on the same recording prefix, then verifies that unusable EDA returns an absent score. See [hardware validation](docs/hardware_validation.md) before connecting a different GSR module or interpreting outputs in older adults.
