"""Extract synchronized release data. Only open original WESAD files for reading."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

RATES = {"EDA": 4, "ACC": 32, "TEMP": 4, "BVP": 64}
SUBJECTS = tuple(i for i in range(2, 18) if i != 12)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rest_interval(counts, samples=480):
    rest = counts[:, 1] == 175
    edges = np.flatnonzero(np.diff(np.r_[False, rest, False]))
    for left, right in edges.reshape(-1, 2):
        if right - left >= samples:
            return int(left), int(left + samples)
    raise ValueError("No complete 120-second resting calibration interval")


def load_subject(source, cache, subject):
    name = f"S{subject}"
    source_pickle = source / name / f"{name}.pkl"
    stat = source_pickle.stat()
    output = cache / f"{name}.npz"
    ledger = cache / f"{name}.json"
    if ledger.exists() and output.exists():
        row = json.loads(ledger.read_text())
        if (row["bytes"], row["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns):
            return row
    row = {"subject": subject, "file": f"{name}/{name}.pkl", "bytes": stat.st_size,
           "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(source_pickle)}
    # Only use the known, locally supplied WESAD research distribution.
    with source_pickle.open("rb") as stream:
        record = pickle.load(stream, encoding="latin1")
    if str(record["subject"]) != name:
        raise ValueError(f"Participant identifier mismatch for {name}")
    wrist = record["signal"]["wrist"]
    labels = np.asarray(record["label"]).ravel()
    n = min(len(labels) // 175, *(len(wrist[k]) // (hz // 4) for k, hz in RATES.items()))
    if not np.isfinite(labels).all() or labels.min() < 0 or labels.max() > 7:
        raise ValueError(f"Unexpected protocol code for {name}")
    lab = labels[:n * 175].reshape(n, 175)
    counts = np.column_stack([(lab == i).sum(axis=1) for i in range(8)]).astype(np.uint8)
    cal = rest_interval(counts)
    row.update(ticks=n, label_samples=len(labels), calibration_start_s=cal[0]/4,
               calibration_end_s=cal[1]/4, label_duration_s=(counts.sum(axis=0)/700).tolist(),
               streams={})
    arrays = {}
    for k, hz in RATES.items():
        filepath = source / name / f"{name}_E4_Data" / (k + ".csv")
        with filepath.open() as stream:
            reader = csv.reader(stream)
            raw_start = float(next(reader)[0])
            actual_rate = float(next(reader)[0])
        if actual_rate != hz:
            raise ValueError(f"Unexpected {name} {k} rate {actual_rate}")
        v = np.asarray(wrist[k])
        row["streams"][k] = {"samples": len(v), "rate": hz, "raw_e4_start_unix": raw_start,
                            "trimmed_tail_samples": len(v)-n*(hz//4),
                            "nonfinite": int((~np.isfinite(v)).sum())}
        value = np.asarray(v[:n*(hz//4)], dtype=np.float32)
        arrays[k.lower()] = value if k == "ACC" else value.ravel()
    del wrist, record, lab, labels
    tmp = cache / f"{name}.{os.getpid()}.part.npz"
    np.savez(tmp, counts=counts, rest=np.array(cal), **arrays)
    os.replace(tmp, output)
    ledger.write_text(json.dumps(row, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/eda/processed/wesad_v1"))
    args = parser.parse_args()
    src, cache = args.source.resolve(), args.cache.resolve()
    if src == cache or src in cache.parents or cache in src.parents:
        raise ValueError("Source and output must be separate directory trees")
    cache.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.name[1:]) for p in src.iterdir() if p.is_dir() and re.fullmatch(r"S\d+", p.name))
    if ids != list(SUBJECTS):
        raise ValueError(f"Unexpected release participant list {ids}")
    rows = []
    for i in ids:
        rows.append(load_subject(src, cache, i))
        print(f"S{i}: {rows[-1]['ticks']/4:.1f}s synchronized; {len(rows)}/{len(ids)}", flush=True)
    tmp = cache / "manifest.tmp"
    tmp.write_text(json.dumps({"subjects":rows, "rates":RATES, "label_hz":700, "eda_hz":4}, indent=2))
    os.replace(tmp, cache / "manifest.json")


if __name__ == "__main__":
    main()
