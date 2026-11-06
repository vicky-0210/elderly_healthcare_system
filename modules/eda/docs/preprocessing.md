# Wrist signal preprocessing and decomposition

The shared training and replay implementation lives in `wearable_eda.preprocessing.wrist`. It operates continuously on synchronized arrays before segmentation. A completed 4 Hz tick can use all native observations in its quarter-second bin. Missing timestamps must be represented as gaps by acquisition. The online adapter requires a regular clock. All model-level scalers belong to training folds.

## Integrity and motion assessment

The E4 signal is in µS. Nonfinite values and conductance outside the configured 0–100 µS plausibility interval are flagged, and the fill uses the last valid measurement. A past 30-second median and MAD of the absolute derivative set a local change threshold above a fixed noise floor. An abrupt change, its direction, local reversal, flatline duration and range violation contribute to artifact confidence. A constant trace lasting 20 seconds receives reduced reliability. The disconnected-contact proxy combines a sustained flatline with a near-zero level. Physical thresholds are heuristics pending measurements on the planned sensor. The data provide neither confirmed disconnections nor annotated saturation.

The ACC pathway converts native E4 counts to g using 64 counts/g. It estimates gravity with a causal low-pass filter, computes magnitude, dynamic acceleration, RMS, magnitude variability and jerk at 32 Hz, then aggregates the trailing second at each completed quarter-second. The movement score combines dynamic RMS and magnitude variability. Motion increases artifact confidence only alongside an unusual EDA step. High motion by itself leaves fast physiological responses intact. Strong artifact confidence attenuates a suspect step, while raw conductance remains a separate view. EDA-only features, cleaning and reliability never consult other streams.

The filter alternatives are no additional low pass, a two-pole 0.65 Hz Butterworth and a two-pole 1 Hz Butterworth. All operate causally at 4 Hz with initialized filter states. The native Nyquist limit is 2 Hz. Explicit EDA gaps of 5 seconds or longer split filter states and exclude every intersecting window. A window containing over 20% missing observations is also excluded.

The quality score is 1 minus artifact confidence. The mean participant fraction of low-quality ticks (score below 0.5) was 0.022% with motion and 0.008% with EDA alone. Across subjects, mean Spearman correlation between movement intensity and absolute EDA change was 0.321. These describe co-occurrence in the released laboratory sessions. Hardware testing with labeled contact loss would be needed to measure artifact sensitivity or specificity.

## Tonic, phasic and residual

| Method | Computation | Time support |
| --- | --- | --- |
| MC-SAED | An asymmetric gated first-order tonic tracker follows relaxation faster than positive excursions. Quality damps suspect tonic updates. A positivity constraint and causal smoother form the phasic estimate. The residual is the cleaned trace minus both components. | Causal after each completed EDA bin |
| cvxEDA | Author package 1.1.0 solves the sparse-driver, Bateman-response and tonic spline QP, with 0.25 s sample interval, 2.0/0.7 s constants, 10 s knots, alpha 0.0008, gamma 0.01. A 120 s core uses 30 s guards where available. Cropping the guard bounds memory; this is a blockwise approximation of whole-record optimization. Inputs stay in µS, without a held-out record-level standardization. | Retrospective comparison |
| Choi | Directly solves equation 1, (I + λ² D₂ᵀD₂) SCL = EDA, with the paper's illustrated λ = 1500. SCR is the signed residual. Applying the same discrete penalty at wrist 4 Hz is an explicit parameter choice. | Retrospective comparison |
| Sparse innovation | A causal 0.035 Hz tonic estimate and nonnegative soft-thresholded phasic innovations driving a 2 s exponential decay. This is an independently specified comparator. | Causal |
| None | Supplies the cleaned trace with a zero phasic placeholder. Model ablations remove component views to isolate the question. | Causal |

Greco et al. describe the reference convex optimization, including its phasic sparsity and tonic regularization. Choi et al. describe the smoothness-prior method in their EDA feature extraction section. [cvxEDA paper](https://doi.org/10.1109/TBME.2015.2474131) · [Author-maintained implementation](https://github.com/lciti/cvxEDA) · [Choi et al., equation 1](https://psi.engr.tamu.edu/wp-content/uploads/2018/01/choi2012titb.pdf)

The fixed continuous methods were measured on all 15 complete recordings. The audit observed an optimal solver status in each of 731 QPs for the reference cvxEDA comparison. The figures below average subject-level measurements. Compute time covers decomposition alone on the host CPU. Workload and hardware affect these times.

| Method | Residual RMSE (µS) | Mean absolute tonic curvature (µS/s²) | Matched cleaned-trace peaks | Median matched amplitude ratio¹ | CPU ms/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC-SAED | 0.01687 | 0.002009 | 75.2% | 0.827 | 0.261 |
| cvxEDA | 0.00663 | 0.001784 | 66.8% | 0.917 | 8.295 |
| Choi | 0 by construction | 0.000127 | 89.7% | 1.024 | 0.247 |
| Sparse innovation | 0.05136 | 0.001032 | 40.0% | 0.600 | 0.014 |

¹ Mean of participant medians. A one-to-one match allows at most 1 s displacement from a cleaned-signal peak of at least 0.03 µS prominence. WESAD has no reference SCL, SCR or artifact annotations. An exact reconstruction can be obtained by defining a component as a residual. Coincidence with an observed deflection provides a morphology proxy, not a physiological ground truth. Alternative thresholds of 0.02 and 0.05 µS are also provided in the [full table](../../../research/results/wesad_loso_v1/preprocessing/decomposition_summary.csv). See the [comparison plot](../../../research/results/wesad_loso_v1/preprocessing/decomposition_comparison.png).

## Calibration and features

Zero calibration preserves absolute measurements. Relative channels have constant zero values. The resting protocol uses median and 1.4826 × MAD of reliable tonic and phasic samples within the explicitly designated 120 seconds. The scale floors are 0.05 µS for SCL and 0.02 µS for SCR. At least 80% of the resting capture must be usable. The calibration filters restart at capture boundaries, so its reference cannot depend on later protocol periods. Absolute and relative channels are both available after calibration. Rolling normalization is provided for prospective experiments but excluded from formal zero/rest LOSO.

The handcrafted set includes tonic location, variability, slope and quartile change, SCR count, peak prominence, mean half-prominence width and rise/recovery halves, energy, positive area and quality proportions. A one-second minimum peak separation and 0.03 µS prominence are used on the native 250 ms grid. Half-width timing is a within-window measure, and windows with no qualifying peak carry zero counts and zero timing summaries. Additional sensors contribute only if requested. Wrist temperature retains its native units and slope. BVP uses a native 64 Hz causal bandpass, RMS amplitude and delayed-confirmation pulse counts to form a trailing 12 s rate proxy. BVP pulse variability under motion can affect this estimator.

## Replay contract

`transform_recording(streams, modalities=..., calibration="zero", decomposition="mcsaed")` returns a time-major float32 matrix and ordered channel names. The low-level research API expects wrist EDA in µS, three-axis E4 ACC counts, temperature in °C and BVP in device units. The high-level predictor accepts physical ACC in g and converts explicitly. To reproduce a time-series decision, pass the synchronized prefix from the session origin with complete bins, extract its last completed window, transpose to channel-major order, and call `window_features`. A window without preceding filter history is a new session. Callers requesting rest calibration supply its two half-open 4 Hz indices and wait until that capture finishes.

Model inputs never include subject identifiers, annotation values, time since session start or phase order. Auxiliary streams are nevertheless potential protocol-motion or environmental cues. E4-to-GSR transfer, sensor mounting, timescale, battery, motion context, and field false alarms require independent validation with older users.
