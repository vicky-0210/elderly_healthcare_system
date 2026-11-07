# OSASUD data contract

Pass the local release pickle to `prepare_data.py --source`. The loader checks the published-layout fingerprint, reads patient, timestamp, physician label and the 80-sample Mindray pleth packet, and preserves the original row index. Source row order, timestamps and quality exclusions are audited for each patient. Each valid section is processed independently. The estimated onset is available at the resolution of the original physician labels. [Dataset descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC9018698/) · [Local release audit](../../../research/results/osasud_v1/source_audit.json)

`prepare_views.py` writes the split manifest locally. In each fold all windows for a person belong to a single partition. Local per-person files and raw timestamps stay outside version control. All four candidate models receive the same scored seconds. [Split and target definitions](../../../research/osasud_protocol.md)
