# IMU tests

`unit/`: loader, validation, label, split, preprocessing and window semantics.

`integration/`: synthetic data preparation, export/inference parity and the module predictor contract.

`fixtures/`: tiny synthetic samples only. Full SisFall data remain local. Hardware integration tests requiring devices belong in the root `tests/hardware/` or use explicit opt-in markers once implemented.

No tests are implemented yet.
