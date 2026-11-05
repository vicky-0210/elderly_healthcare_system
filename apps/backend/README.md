# Backend/API

Python package shell: `src/wearable_api/`.
- `routes/`: request handlers.
- `services/`: application operations and prediction ingestion.
- `schemas/`: API validation referencing shared contracts.
- `storage/`: persistence adapters.
- `tests/`: API unit/integration tests.

Select the web framework and persistence implementation when building the service. The API receives prediction and device-health events; it must not import model training or offline dataset processing. Shared contracts live at repository `contracts/`. No runnable API is supplied.
