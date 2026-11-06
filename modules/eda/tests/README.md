# EDA verification

From the repository root, with pytest and the module installed, run `python -m pytest -q modules/eda/tests`. The tests exercise held-out subject isolation, frozen scaling, signal preprocessing, model graphs, checkpoint reload, metric definitions and the lightweight selection rule. The fixtures are synthetic.

The measured-data verification commands live in the [EDA runbook](../README.md). They reconstruct window identities, re-score each fold's predictions and replay the all-subject exports on synchronized native-rate input.
