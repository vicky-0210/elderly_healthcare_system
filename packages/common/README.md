# Common utilities

Python package: `wearable_common`.

`inference.py` defines only the external predictor interface. `config/`, `io/` and `logging/` reserve genuinely reusable helpers. Versioned payload contracts are documented in repository `contracts/`.

Keep label mapping, filters, decomposition, window rules, physiological thresholds and neural networks in the owning signal module. Common code must remain importable without any signal's training framework.

The initial interface is abstract and does not implement prediction. Generic utilities and tests are pending.
