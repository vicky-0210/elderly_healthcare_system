# Data and experiment provenance

Original datasets stay local and immutable. Preserve source version, file checksums, record identity, patient/subject identity, clock information, units and channel mapping in local manifests. Persist exclusion reasons and missing-data masks.

Partition by patient/subject before augmentation or overlapping window generation. Preserve official PTB-XL folds; generate and persist grouped splits for OSASUD, SisFall and WESAD. Fit scaling, decomposition parameters, feature selection and calibration without test leakage.

Do not commit patient-level manifests or wearable recordings. Shared documentation may contain aggregate statistics without identities. Small test fixtures should be synthetic and labeled as such.

Do not assume dataset sensor units, analog response or placement match the wearable. Keep device validation separate in each module's `docs/hardware_validation.md` and local `data/<signal>/wearable/`.

Every exported model must identify its preprocessing, label map and input contract. Research checkpoints alone are not deployment artifacts.
