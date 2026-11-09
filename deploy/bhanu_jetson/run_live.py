"""Live detector on the Jetson Orin Nano with a MAX30102. Usage: python3 run_live.py [--bus 7] [--log events.csv]
Prints one line every 10 s and appends it to the CSV log."""
import argparse
import csv
import time

import numpy as np

from max30102 import MAX30102
from runtime import Detector, ppg_second_ok

ap = argparse.ArgumentParser()
ap.add_argument("--bus", type=int, default=7)
ap.add_argument("--log", default="events.csv")
a = ap.parse_args()
sensor, det, state = MAX30102(a.bus), Detector("."), {}
with open(a.log, "a", newline="") as fh:
    w = csv.writer(fh)
    print("warming up: first decision after 65 s, context features need about 30 min to settle", flush=True)
    while True:
        spo2, ir = sensor.read_second()
        r = det.push(spo2, ppg_second_ok(ir, state))
        if r is not None:
            msg = "no decision (bad signal)" if not r["ok"] else (
                f"score {r['score']:.2f} cnn {r['p_cnn']:.2f} gbm {r['p_gbm']:.2f} -> {'EVENT' if r['alarm'] else 'normal'}")
            print(time.strftime("%H:%M:%S"), msg, flush=True)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), r["ok"], r["p_cnn"], r["p_gbm"], r["score"], r["alarm"]]); fh.flush()
