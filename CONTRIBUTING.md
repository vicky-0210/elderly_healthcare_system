# Contributing

## Ownership boundaries

| Area | Primary responsibility |
| --- | --- |
| `modules/ecg`, `modules/ppg`, `modules/imu`, `modules/eda` | Corresponding signal team member |
| `acquisition` | Sensor/firmware team |
| `edge`, `deploy/jetson` | Edge integration team |
| `apps/backend` | API team |
| `apps/frontend` | Frontend team |
| `packages/common`, `contracts` | Shared review by affected owners |

Keep independent module changes in their own folders. Agree on contract changes before integrating them. A directory shared by all signals is not a mandate to share physiology-specific code.

## Changes

- Commit code, config templates, documentation and small synthetic fixtures.
- Keep datasets, identifiable recordings, credentials, patient-level split manifests, checkpoints and generated reports out of Git.
- Never overwrite another teammate's local path configuration.
- Add a dependency to the owning package/environment, not to every module. Pin and validate real dependencies when implementations are introduced.
- Implement module-level unit/integration tests alongside functionality; use root tests for system boundaries.
- Preserve patient/subject separation before creating overlapping windows or augmented copies.
- Record data version, split, seed, preprocessing version and label mapping with every experiment.
- Review model changes against both offline and actual sensor-domain requirements.

## Before committing

Inspect `git status --short` and `git diff --cached --stat`. Confirm that no local data, weights, recordings, secrets or generated artifacts are staged. Update affected docs and add meaningful tests for implemented behavior.

This scaffold contains no remote configuration and performs no commits.
