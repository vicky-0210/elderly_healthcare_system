# ECG research and inference workflow

The implemented experiment is configured in `experiments/configs/lvh_comparison.json` and launched with `scripts/run_workflow.py`. See the module README for exact commands.

The source dataset is immutable. Source checksums, calibrated units, lead names, patient identities, folds and duplicates are audited. Physical 500-Hz Lead II is resampled to 360 Hz; derived 10-second signals, features and preprocessing controller live in the ignored local cache.

The primary wearable models are single-lead CNN, CNN–Transformer and CNN–diagonal-SSM. Each retains a fixed-mV voltage route alongside a per-window shape route. Feature scaling is fitted on training patients and stored in the checkpoint. Labels, demographics and diagnostic reports are excluded from neural inputs. Age/sex and quality-only controls are separate experiments.

Development training uses official folds 1–6 and epoch selection uses 7–8. Each model is freshly refitted on 1–8 for its selected number of epochs. Fold 9 fits positive-slope logistic calibration and frozen operating thresholds. Fold 10 opens only after every configuration/threshold is saved. One cluster bootstrap draw resamples patients and keeps all their recordings together. Seed variation is reported separately from conditional test-sampling uncertainty.

The proposed preprocessing is an experimental controller over conservative correction settings. Its automatic landmarks measure engineering proxies; it does not establish clinical QRS/ST fidelity. Independent development morphology reports and conventional/minimal/SWT comparisons are saved. The main architecture comparison uses the same constrained candidate for every model. It is not the full classifier-level preprocessing or distillation ablation program.

Runtime `ECGPredictor` accepts a complete, calibrated 10-second Lead II window with explicit units and sample rate. It checks integrity, applies the saved controller and feature extraction, then returns a calibrated research score and quality status. Missing calibration, gaps and ADC-clipped input cause an insufficient-signal result. Unknown electrode placement or analog response still requires hardware-domain study; software checks cannot verify them.

Preserve float64 precision when converting raw ADC samples to physical mV and through resampling. Narrow to float32 at the canonical model input. Pre-rounding a raw window to float32 can move a near-flat sample across the quality-counter tolerance; the numerical parity check uses original WFDB physical values and reports lower-precision differences separately.

The exported ONNX receives preprocessed waveform and matching features. The Python bundle contains preprocessing, model and calibration. Diagonal-SSM FFT kernels are converted into equivalent fixed-length grouped convolutions for export; training-window parity is measured. Physical Nano, TensorRT FP16 and end-to-end hardware latency remain unverified.

Cross-lead distillation, externally pretrained models and clinical validation are separate future protocol stages; the current supervised runs must not be described as completing them.
