# EDA/GSR pipeline

Reserve separate tonic/phasic processing and optional accelerometer-based motion handling. Exclude undefined, amusement and meditation intervals, and define how windows crossing boundaries are rejected. Fit transformations on training subjects only. Model architecture and representation remain this module's decision.

Offline stages: load -> validate -> construct/audit group splits -> map labels and segment by the documented policy -> fit preprocessing on training data -> train -> evaluate -> export.

Runtime stages: validate the live window -> apply matching preprocessing -> predict -> produce status/scores using the shared contract. The order of label alignment, gap handling and windowing remains signal-specific; do not use a universal pipeline merely because folders share names.

Implement training and evaluation entry points in `scripts/` with reusable logic under `src/`. Store fitted transformations with the export. Architecture/window choices and measurements are pending, not supplied defaults.
