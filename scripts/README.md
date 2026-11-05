# Repository operations

Reserved directories:
- `dev/`: local setup and developer tooling.
- `data/`: generic manifest/checksum utilities; keep dataset parsing module-specific.
- `validation/`: repository/integration validation tools.
- `release/`: artifact verification and release packaging.

Module prepare/train/evaluate/export entry points live in `modules/<signal>/scripts/`. This directory contains no automated dataset downloads or Git commit hooks.
