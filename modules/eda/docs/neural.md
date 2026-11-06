# WESAD neural experiments

All networks make an end-of-window prediction from 60 seconds of E4 wrist data. Four consecutive native 4 Hz observations are embedded as one second of temporal context. The short sequence and hidden width of 16 keep each candidate small enough for a practical CPU comparison.

## Reproduction

Run from the repository root in the EDA research environment after preparing the WESAD cache.

```powershell
python modules/eda/scripts/train_neural.py --list
python modules/eda/scripts/train_neural.py --device cuda
python modules/eda/scripts/audit_neural.py
```

Use `--models cnn_zero pg_zero` to run a subset or `--run PATH` to select another output directory. Completed subject checkpoints support resuming the same configuration. Select a fresh run directory after changing the data, architecture, implementation or training budget.

The run directory defaults to `artifacts/eda/runs/wesad_loso_v1`. Within `neural/<experiment>/Sxx/`, `model.pt` contains the 14-subject refit, scaler, temperature, input contract and architecture, `history.csv` contains development and refit curves, and `inner_predictions.csv` contains the development-model outputs on two validation subjects. `predictions.csv` holds predictions on the single outer test subject. The experiment directory combines all 15 held-out predictions and fold choices.

## Subject protocol and optimization

Each participant is windowed independently. Only subject IDs determine the outer, inner and refit memberships. For each held-out subject, offsets +1 and +8 in the circular numerically sorted participant roster select two inner-validation subjects. The other 12 subjects fit numeric medians and IQRs, inverse-frequency class weights and neural weights. Training quantiles use one native observation per second in the training windows. Quality, artifact and missingness indicators retain their original scales. Transformed inputs are clipped at ±30. No test windows contribute to these operations. Zero calibration and the designated two-minute resting calibration are evaluated separately on matched scored windows.

Model width is 16, dropout 0.15, AdamW learning rate 0.002, weight decay 0.001, maximum gradient norm 2 and minibatch size 256. The epoch budget is six with patience three. The development epoch maximizes the mean of the two subject-specific macro F1 scores, with validation cross-entropy breaking ties. A bounded scalar temperature is fitted from the chosen development checkpoint's validation logits. A fresh network with the same fold seed then fits all 14 remaining subjects for the selected number of epochs, using newly fitted scaling and class weights. The inner temperature is transferred to that refit, so calibration on new subjects remains an empirical outcome. The fixed-budget PG cosine schedule runs from 0.002 to 0.0002 across six epochs. A separate PG focal-loss experiment uses exponent two.

The small search budget makes the comparisons reproducible on the available GPU. Larger model widths, independent seed repetitions and longer training may change rankings. PyTorch seeds and versions are recorded, while precise floating-point results can vary across devices and library versions.

## Architectures and comparisons

- Standalone networks cover CNN, LSTM, BiLSTM, GRU, BiGRU, stacked dilated TCN, parallel multi-scale TCN and a two-head positional Transformer.
- Hybrids include CNN–BiLSTM, CNN–Transformer, CNN–TCN–attention, TCN–attention, multi-scale TCN–GRU/Transformer, independent tonic/phasic branches, cross-branch attention, gated multi-scale attention, reliability-weighted fusion and engineered-feature MLP fusion.
- PG-MSTN combines a tonic TCN with dilations 1/2/4, a phasic parallel TCN with dilations 1/4/12, a motion/quality embedding and a handcrafted-feature MLP. Quality enters the learned gate and a log prior on the two EDA branch weights. The default temporal head learns softmax attention pooling.
- Four fusion controls use concatenation, learned global weights, a sample-dependent gate and cross-attention. Other paired controls vary recurrent/Transformer aggregation, single-scale backbone, handcrafted features, focal loss and use of relative components. Compare `pg_no_components_no_hand_zero` against `pg_no_hand_zero` to isolate the contribution of the decomposed channels with no feature side channel. Compare `pg_eda`, `pg`, `pg_temp` and `pg_fullwrist` in the same calibration protocol for incrementally richer sensors. `pg_intrinsic_quality_zero` retains ACC features and uses EDA-only artifact assessment.
- Masked pretraining reconstructs raw, tonic and phasic samples from aligned masked multiview patches; supervised fine-tuning uses the same TCN as the scratch control. Contrastive pretraining uses two gain-perturbed views with short aligned masks. Other overlapping windows from the same subject are excluded from in-batch negatives. Both pretraining objectives use two epochs over windows from permitted training participants only. The pool is limited to windows eligible under the four-state purity rule, and state identities are discarded in SSL. SSL adds compute relative to the scratch controls.
- The learned physics model starts with continuously computed components and applies a shared pointwise refiner. Applying this block before segmentation produces the same components as applying it to slices. A bounded residual, nonnegative SCR and a nonnegative SCL accompany the supervised objective. The auxiliary loss combines normalized reconstruction L1, a second-difference SCL penalty, residual magnitude and phasic magnitude. The `physics_no_aux_zero` control retains the same neural refiner architecture. `pg_cvxeda_zero` supplies retrospective cvxEDA components to the ordinary PG backbone; `pg_cv_physics_zero` supplies them to the learned refiner. Upper-layer handcrafted features and derivatives are from the continuous preprocessor. The refiner updates SCL/SCR inputs to the network and does not recompute those extra features.
- The three- and four-state PG runs evaluate separately from binary comparisons. Their encoded state order appears in `meta.json`. Within-window bidirectional models use both past and later samples inside a completed window.

## Verification and latency

The audit reloads every participant checkpoint on its training device, checks window identities, and compares its probabilities with the saved outer predictions to absolute tolerance 0.0001. It separately records the largest CPU probability difference and number of changed decisions. The smallest-ID participant's checkpoint supplies the batch-one CPU timing, including the frozen scaler and temperature, after ten warmups and across 50 timed calls with one thread. The same representative median and p95 are repeated across the architecture's fold rows. Parameter and buffer bytes, input bytes and checkpoint size are reported separately. Target-device acquisition, runtime memory and latency require hardware measurements.

Three follow-up experiments raised only the epoch limit to 18 for EDA-only multi-scale TCN, PG with motion and PG with full-wrist inputs. They retained the primary seeds, grouped validation, early-stopping patience and zero-calibration scoring windows. They appear under `budget_sensitivity` in the complete comparison, outside the original selection pool.

The audit writes `neural_audit/host_inference.csv`, pooled learning histories, a learning-curve figure, a held-out decomposition figure and a per-subject decomposition table. For a full-data research refit, `wearable_eda.training.neural_suite.export_full` uses the median inner-selected epoch and temperature. The resulting artifact serves inference through `wearable_eda.inference.neural.NeuralPredictor`. Accuracy estimates come from the segmented LOSO checkpoints.

Eligible arousal states refer to the WESAD laboratory protocol. Clinical stress, heat exposure, real-world movement and changes in sensor contact require external validation.
