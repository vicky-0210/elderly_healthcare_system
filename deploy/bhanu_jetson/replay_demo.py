"""Run the detector on a CSV of per-second SpO2 (column 'spo2', optional 'ppg_ok') without any sensor - use it to test the install.
Usage: python3 replay_demo.py sample_spo2.csv"""
import sys
import time

import numpy as np
import pandas as pd

from runtime import Detector

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_spo2.csv")
ok = df["ppg_ok"].astype(bool).values if "ppg_ok" in df else np.ones(len(df), bool)
det, n, t0 = Detector("."), 0, time.time()
for sp, o in zip(df["spo2"].values, ok):
    r = det.push(sp, o)
    if r is not None:
        n += 1
        print(f"t={r['t_end']:6d}s", "no decision" if not r["ok"] else f"score {r['score']:.2f} -> {'EVENT' if r['alarm'] else 'normal'}")
print(f"{n} decisions for {len(df)} s of data in {time.time() - t0:.1f} s")
