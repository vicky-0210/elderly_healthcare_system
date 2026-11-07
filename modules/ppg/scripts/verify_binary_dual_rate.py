"""Reconstruct targets, scaler and checkpoint scores for every new fold."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.datasets.osasud import RESULT
from wearable_ppg.evaluation.metrics import patient_ap
from wearable_ppg.models.dual_rate import VARIANTS, make_model
from wearable_ppg.training.dual_rate import REPORT_DIR, RUN_DIR, load_data, window_batch
from wearable_ppg.training.study import norm_fit, seconds


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tol = 3e-4 if device.type == "cuda" else 5e-3
    frame, y, t, ids, ends, start = load_data()
    start = torch.tensor(start, device=device)
    folds = pd.read_csv(RESULT / "folds.csv")
    verified = blocks = 0
    largest_error = 0.
    ap_rows = {name: [] for name in VARIANTS}
    for k in range(5):
        test_ids = set(folds.loc[folds.fold == k, "patient"])
        val_ids = set(folds.loc[folds.fold == (k+1)%5, "patient"])
        train_ids = set(folds.patient)-test_ids-val_ids
        train = ends[np.isin(ids[ends], list(train_ids))]
        center, scale = norm_fit(frame, train)
        x = torch.asinh((torch.tensor(frame, device=device)-torch.tensor(center, device=device))
                        /torch.tensor(scale, device=device))
        for name in VARIANTS:
            folder = RUN_DIR / name / f"fold{k}"
            meta = json.loads((folder / "done.json").read_text())
            assert set(meta["train"]) == train_ids and set(meta["test"]) == test_ids and set(meta["val"]) == val_ids
            with np.load(folder / "scaler.npz") as saved:
                np.testing.assert_array_equal(saved["center"], center)
                np.testing.assert_array_equal(saved["scale"], scale)
            net = make_model(name).to(device).eval()
            net.load_state_dict(torch.load(folder/"model.pt", weights_only=True, map_location=device))
            for split, people in (("test",test_ids), ("validation",val_ids)):
                selected = ends[np.isin(ids[ends],list(people))]
                targets = seconds(selected).ravel()
                with np.load(folder / f"{split}.npz") as saved:
                    for key, expected in (("y",y[targets]), ("t",t[targets]), ("patient",ids[targets]),
                                          ("arrival",np.repeat(t[selected]+1,5))):
                        np.testing.assert_array_equal(saved[key],expected)
                    logits = saved["logit"]
                    a,b = meta["calibration"]
                    probs = expit(a*logits+b)
                    np.testing.assert_allclose(saved["p"], probs, atol=1e-6)
                    np.testing.assert_array_equal(probs>=meta["threshold"], saved["decision"])
                    ap = patient_ap(y[targets], probs, ids[targets])
                    if split == "test":
                        ap_rows[name].append(ap)
                    else:
                        np.testing.assert_allclose(ap, meta["val_patient_ap"])
                    native = np.array_split(np.arange(len(selected)), int(np.ceil(len(selected)/512)))
                    for ix in (native[0], native[len(native)//2], native[-1]):
                        input_tensor = window_batch(x, start, selected[ix], net.context)
                        with torch.inference_mode():
                            replay = net(input_tensor)[:, :, 0].cpu().numpy()
                        expected = logits.reshape(-1,5)[ix]
                        largest_error = max(largest_error, float(abs(replay-expected).max()))
                        np.testing.assert_allclose(replay,expected,atol=tol)
                        blocks += len(ix)
            verified += 1
    summary = pd.read_csv(REPORT_DIR / "comparison.csv").set_index("model")
    for key, scores in ap_rows.items():
        np.testing.assert_allclose(np.mean(scores), summary.loc[key,"patient_ap"])
    print(f"{verified} checkpoints, {2*verified} score files, {blocks:,} replayed blocks; max logit difference {largest_error:.7g} ({device.type}).")
    print("All saved labels, timestamps, group assignments, scalers, decisions and patient AP checks passed.")


if __name__ == "__main__":
    main()
