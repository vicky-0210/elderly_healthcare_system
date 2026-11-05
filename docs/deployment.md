# Deployment preparation

`deploy/jetson/` owns the target environment and profiling; `deploy/backend/` and `deploy/frontend/` own service packaging. `deploy/services/` reserves process-management definitions, and `deploy/monitoring/` reserves logging/health configuration.

No Dockerfiles, system services or infrastructure are supplied as working deployments yet. Add them after selecting and verifying real dependencies.

A release requires compatible exported models, versioned input contracts, preprocessing parity, explicit unavailable-signal behavior and on-device resource measurements. Validate queue backpressure, restart/reset behavior, clock discontinuities and independent signal failures.

The Nano should load only required inference dependencies. API/frontends should receive structured prediction and health events. Raw physiological recordings require an explicit storage and access design; they are not frontend fixtures.
