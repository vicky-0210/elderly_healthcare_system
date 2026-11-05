# WESAD data contract

Use wrist EDA from Empatica E4 at 4 Hz, not a chest signal with a different sample rate. Keep source labels 0..4, using only baseline (1) and stress (2) for the initial binary task. Align the annotation clock to the wrist signal; do not assume identical array rates or lengths.

Check subject identity, wrist stream selection, 4 Hz timing, annotation alignment, included/excluded states, gaps, saturation and subject-independent splits. Do not describe laboratory label classification as clinical stress diagnosis.

Local input paths are set in repository `configs/paths.local.json`; schema and mapping decisions belong in this module's `configs/dataset.json`. Do not commit local source files, record identifiers or split manifests.
