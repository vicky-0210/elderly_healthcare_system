# Generated model artifacts

Each signal has its own `checkpoints/`, `exports/`, `runs/` and `reports/`. These directories contain local generated payloads and are ignored by Git. Use unique run/version subdirectories instead of overwriting a teammate's output.

An export must eventually include model weights plus a manifest describing:
- signal, intended task, model and preprocessing versions;
- exact input channel order, units, sampling rate, timing and window requirements;
- label mapping and threshold/calibration provenance;
- dataset version, patient/subject split reference and training seed;
- quality/rejection policy and missing-data handling;
- file checksums, inference format and precision;
- measured hardware compatibility and limitations.

No trained model or inference-ready manifest is supplied. Register only validated exports in the local model registry. Keep software versioning in Git; choose separate artifact storage with the team when needed.
