# SisFall data contract

SisFall is sampled at 200 Hz. Preserve subject and trial/activity identity. Explicitly select one accelerometer and one gyroscope stream for the six-axis wearable experiment; do not assume all raw columns match the MPU-9250 configuration. Document raw units and conversion factors.

Check subject IDs, activity mapping, channel order, units, saturation, sample count/rate and subject-disjoint splits. Do not report exact onset accuracy without appropriate additional time annotations.

Local input paths are set in repository `configs/paths.local.json`; schema and mapping decisions belong in this module's `configs/dataset.json`. Do not commit local source files, record identifiers or split manifests.
