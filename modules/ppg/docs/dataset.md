# OSASUD data contract

The distributed Mindray PPG stream is 80 Hz. Preserve patient identity and timestamps. Keep all five source label names, including the apnea subtypes. Select binary, three-class or five-class processing explicitly instead of silently merging events. Inspect source fields and annotation timing rather than assuming a file layout.

Check timestamps, monotonic segments, missing samples, units, optical channels, event intervals, label vocabulary and patient independence. Document any excluded gaps or interrupted recordings.

Local input paths are set in repository `configs/paths.local.json`; schema and mapping decisions belong in this module's `configs/dataset.json`. Do not commit local source files, record identifiers or split manifests.
