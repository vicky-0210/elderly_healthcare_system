# PTB-XL data contract

Use the exact LVH diagnostic statement, not the HYP superclass. Metadata is not anatomical LVH confirmation. Keep all recordings of a patient in the official folds, including all derived windows. Freeze normal-control exclusions and confidence rules before training. Retain 12-lead research/teacher experiments separately from the single-lead wearable model.

Check waveform/metadata linkage, units, lead names, missingness, clipping, diagnosis contradictions, duplicate recordings and official patient-fold independence.

Local input paths are set in repository `configs/paths.local.json`; schema and mapping decisions belong in this module's `configs/dataset.json`. Do not commit local source files, record identifiers or split manifests.
