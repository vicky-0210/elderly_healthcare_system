# Local data workspace

Raw public datasets may live anywhere on each teammate's machine; configure their roots in `configs/paths.local.json`. The `raw/` directories here are optional and do not require copying existing datasets.

For each signal:
- `raw/`: original data, never modified in place.
- `interim/`: validated/imported working data.
- `processed/`: derived arrays and windows.
- `splits/`: local patient/subject split manifests and provenance.
- `wearable/`: real hardware captures, calibration sessions and replay inputs.

Payloads are ignored. Only README files and .gitkeep placeholders are tracked. Do not add patient identities to committed examples. Tests should use tiny synthetic fixtures, not recordings copied from datasets.
