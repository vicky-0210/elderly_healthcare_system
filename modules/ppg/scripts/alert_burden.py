"""Exploratory high-specificity operating point for the selected model."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.datasets.osasud import RUN, RESULT
from wearable_ppg.evaluation.compare import read_scores
from wearable_ppg.evaluation.metrics import event_rows


def counts(data, threshold):
    rows, _ = event_rows(data["y"], data["p"] >= threshold, data["t"], data["patient"], data["arrival"])
    return np.array([sum(row[key] for row in rows) for key in ("event_tp", "event_fp", "event_fn")])


def main():
    winner = json.loads((RESULT / "selection.json").read_text())["winner"]
    output = []
    for k in range(5):
        folder = RUN / winner / f"fold{k}"
        val, test = (read_scores(folder, split) for split in ("validation", "test"))
        hours = len(val["y"]) / 3600
        thresholds = np.r_[np.quantile(val["p"], [0.9, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9999]), 1.01]
        candidates = []
        for threshold in thresholds:
            tp, fp, fn = counts(val, threshold)
            if fp / hours <= 1:
                candidates.append((tp, -fp, threshold))
        tp, neg_fp, threshold = max(candidates)
        held_tp, held_fp, held_fn = counts(test, threshold)
        output.append(dict(fold=k, threshold=threshold, validation_tp=tp, validation_fp_per_h=-neg_fp/hours,
                           test_tp=held_tp, test_fp=held_fp, test_fn=held_fn, test_hours=len(test["y"])/3600))
    df = pd.DataFrame(output)
    df.to_csv(RESULT / "alert_burden.csv", index=False)
    totals = df.sum()
    print(f"Held-out recall {totals.test_tp/(totals.test_tp + totals.test_fn):.4f}, "
          f"{totals.test_fp/totals.test_hours:.3f} false events/hour, {totals.test_tp:.0f} matched events.")


if __name__ == "__main__":
    main()
