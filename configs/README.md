# Shared configuration

- `paths.example.json`: copy to `paths.local.json`, which is ignored. Dataset roots are teammate-specific absolute paths. Relative workspace outputs resolve from repository root.
- `model_registry.example.json`: copy to `model_registry.local.json` for local integration. No model is enabled by default.

Signal-specific parameters live in `modules/<signal>/configs/`. Device configuration lives in `acquisition/configs/`; scheduling lives in `edge/configs/`.

Templates document intended fields. A config loader/validator is not implemented yet. Null values mean unresolved, not automatic defaults; dependent operations must fail rather than guess. Keep shared inference contracts separate from per-signal algorithms.
