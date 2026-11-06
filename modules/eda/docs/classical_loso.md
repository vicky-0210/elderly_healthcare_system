# Classical WESAD experiments

These experiments classify windows by their WESAD protocol condition. The binary positive class means stress-associated physiological arousal during the laboratory protocol.

## Reproduce and verify

From the repository root, after preparing the WESAD cache and installing the EDA research environment, run these commands in PowerShell.

```powershell
.\.venv\Scripts\python.exe modules/eda/scripts/train_classical.py --list
.\.venv\Scripts\python.exe modules/eda/scripts/train_classical.py --jobs 4
.\.venv\Scripts\python.exe modules/eda/scripts/verify_classical.py
.\.venv\Scripts\python.exe modules/eda/scripts/benchmark_classical.py
```

Use `--experiment logistic__eda__zero` with either training or verification for a quick single-model check. Both commands default to the derived cache in `data/eda/processed/wesad_v1` and the run directory in `artifacts/eda/runs/wesad_loso_v1/classical`. An interruption can be resumed with the same command. Data, dependency and source fingerprints protect completed folds from accidental mixing. Use a new `--output` for changed experiments. The verification command reloads each participant's saved estimator, checks group membership, reconstructs probabilities on the held-out participant, and compares them with the CSV.

## What gets compared

| Study | Design |
| --- | --- |
| Binary candidates | Logistic regression, RBF SVM, random forest, XGBoost and LightGBM. Each uses four nested input sets, starting with EDA and adding ACC, temperature and then BVP. Both zero and initial-rest calibration are evaluated separately. |
| Preprocessing | Paired random forest controls vary one setting at a time. Comparators cover unfiltered, conservative 0.65 Hz and 1 Hz filtering, no decomposition, Choi, sparse, cvxEDA and MC-SAED decomposition, plus motion-conditioned, EDA-only and disabled quality assessment. All five classifiers also run on EDA-only cvxEDA features. |
| Windowing | The reference uses 60 s windows, a 15 s stride and 80% label purity. Matched controls change the window to 30 or 90 s, the stride to 30 s, or purity to 100%. |
| Streaming sensitivity | A causal rolling normalization control uses recent signal history, including previous evaluation-period samples. It is reported separately from formal zero and initial-rest calibration. |
| Negative controls | On the same eligible windows, random forests see only the ACC-derived or only the temperature-derived columns. Strong performance here can indicate information about activity or the laboratory protocol. |
| Secondary tasks | All five classifiers use EDA-only zero-calibration input for three-class baseline/stress/amusement and four-class baseline/stress/amusement/meditation. |

## Fold construction

The split is defined by participant identity before model fitting. All signal and window transformations run independently inside each participant's synchronized session. For every held-out participant, two other participants form the inner validation set. Their positions are fixed at offsets +1 and +8 in the circular sorted list of 15 participants. Fit each candidate setting on the remaining 12. Training partitions alone determine median imputation, feature median/IQR and balanced class weights.

| Estimator | Inner search |
| --- | --- |
| Logistic regression | C = 0.1 or 1. L-BFGS for binary labels and float64 Newton–Cholesky for the three- and four-class fits. A fit that reaches its iteration cap raises an error. |
| RBF SVM | C = 1 or 10, gamma = scale |
| Random forest | 80 trees, minimum leaf size 3 or 10 |
| XGBoost | 80 histogram boosting rounds, learning rate 0.075, depth 2 or 4 |
| LightGBM | 80 boosting rounds, learning rate 0.075, 7 or 15 leaves |

Select the setting with the highest unweighted mean of the two validation participants' macro F1. Ties follow the listed grid order. Fit a positive temperature between 0.2 and 5 by minimizing participant-balanced negative log likelihood on the two inner validation subjects. Refit the selected estimator and its preprocessing on all 14 training subjects. Apply the inner temperature to its held-out logits. Binary SVMs use the decision margin and multiclass SVMs use the one-vs-rest decision scores. The inner temperature is applied to the refitted estimator, so held-out Brier scores provide a necessary check on calibration transfer.

## Outputs

Each experiment has `meta.json`, `folds.csv`, `predictions.csv` and a `subjects/S*/` directory. `meta.json` records the WESAD label mapping, signals, preprocessing, versions, seed and provenance hashes. For binary tasks, `p0` is baseline and `p1` is stress. The secondary tasks append amusement as `p2` and meditation as `p3`. The per-fold records include both searched scores, group assignments, temperature, fitted model size and CPU batch-one classifier timing. Latency excludes acquisition, window accumulation and signal preprocessing. Tree counts and support-vector counts describe the complexity of estimators without a directly comparable neural parameter count.

Only load joblib checkpoints produced by a trusted run. The final deployment export refits the selected configuration on all 15 participants. It uses the most frequently selected inner-fold hyperparameters and the geometric mean of the 15 inner temperatures. Use the saved 14-participant checkpoints for the reported LOSO evaluation.

`cpu_benchmark.csv` compares one configuration per classical family, chosen by mean zero-calibration inner score. A fresh process loads the S2 fold model and repeatedly classifies the same held-out S2 feature row. Each process uses one thread, 20 warm-up predictions and 300 timed predictions. Whole-process warm and peak resident memory include Python and imported libraries. The delta is measured against the process before loading its checkpoint. The timing excludes raw-signal feature extraction and represents the development workstation.
