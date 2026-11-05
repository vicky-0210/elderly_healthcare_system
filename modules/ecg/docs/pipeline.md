# ECG pipeline

Retain physical voltage calibration. Evaluate baseline removal and denoising for QRS and ST/T distortion. Export fitted scaling and preprocessing metadata with the model. A 12-lead-trained model must not be silently fed one lead. Keep normalization choices and single-lead experiments explicit.

Offline stages: load -> validate -> construct/audit group splits -> map labels and segment by the documented policy -> fit preprocessing on training data -> train -> evaluate -> export.

Runtime stages: validate the live window -> apply matching preprocessing -> predict -> produce status/scores using the shared contract. The order of label alignment, gap handling and windowing remains signal-specific; do not use a universal pipeline merely because folders share names.

Implement training and evaluation entry points in `scripts/` with reusable logic under `src/`. Store fitted transformations with the export. Architecture/window choices and measurements are pending, not supplied defaults.
