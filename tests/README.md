# System tests

- `contracts/`: interchange schema and predictor interface checks.
- `integration/`: acquisition, edge, modules and API boundaries.
- `end_to_end/`: synthetic whole-system scenarios.
- `hardware/`: opt-in real sensors and actual Jetson validation.
- `fixtures/`: small synthetic shared fixtures only.

Signal-specific tests live in `modules/<signal>/tests/`. No tests require private local datasets by default. There are no implemented system tests yet.
