"""Reconstruct score alignment from patient packets and exercise every saved checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.datasets.osasud import CACHE, RESULT, RUN
from wearable_ppg.datasets.windows import load_views
from wearable_ppg.models.respiratory import PRIMARY, CONTROLS, RespiratoryNet
from wearable_ppg.inference.predictor import PPGPredictor
from wearable_ppg.training.study import norm_fit


def verify(source=None):
    audit = json.loads((RESULT / "source_audit.json").read_text())
    if source:
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == audit["source_sha256"]
    folds = pd.read_csv(RESULT / "folds.csv").set_index("patient")
    comparison = pd.read_csv(RESULT / "comparison.csv").set_index("model")
    x, y, ts, ids, cached_ends = load_views()
    endpoints = []
    offset = 0
    for patient in folds.index:
        with np.load(CACHE / f"p{patient:02}.npz") as source_rows:
            t, labels, ok = (source_rows[key] for key in ("t", "y", "valid"))
            np.testing.assert_array_equal(t, ts[offset:offset + len(t)])
            np.testing.assert_array_equal(labels, y[offset:offset + len(t)])
            streak = 0
            for j, usable in enumerate(ok):
                if j and t[j] != t[j - 1] + 1:
                    streak = 0
                streak = streak + 1 if usable else 0
                if streak >= 30 and (streak - 30) % 5 == 0:
                    endpoints.append(offset + j)
            offset += len(t)
    ends = np.array(endpoints)
    np.testing.assert_array_equal(ends, cached_ends)
    assert offset == audit["rows"]
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CPU kernels can differ slightly from the training workstation's CUDA convolutions.
    tolerance = 3e-4 if device.type == "cuda" else 5e-3
    max_difference = 0.0
    verified = 0
    for name in PRIMARY + CONTROLS:
        patient_aps = []
        for k in range(5):
            path = RUN / name / f"fold{k}"
            meta = json.loads((path / "done.json").read_text())
            groups = dict(test=folds.index[folds.fold == k], val=folds.index[folds.fold == (k+1)%5])
            groups["train"] = folds.index.difference(groups["test"].union(groups["val"]))
            for split, members in groups.items():
                assert sorted(meta[split]) == sorted(members)
            training = ends[np.isin(ids[ends], groups["train"])]
            center, scale = norm_fit(x, training)
            with np.load(path / "scaler.npz") as saved:
                np.testing.assert_array_equal(saved["center"], center)
                np.testing.assert_array_equal(saved["scale"], scale)
            model = RespiratoryNet(name).to(device).eval()
            model.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
            for split, members in (("validation", groups["val"]), ("test", groups["test"])):
                sel = ends[np.isin(ids[ends], members)]
                delay = 10 if name == "lag10" else 0
                target = (sel[:, None] + np.arange(-4, 1) - delay).ravel()
                with np.load(path / f"{split}.npz") as data:
                    for key, expected in (("patient", ids[target]), ("t", ts[target]), ("y", y[target]),
                                          ("arrival", np.repeat(ts[sel] + 1, 5))):
                        np.testing.assert_array_equal(data[key], expected)
                    p = data["p"]
                    a, b = meta["calibration"]
                    np.testing.assert_allclose(p, expit(a * data["logit"] + b), atol=1e-6)
                    np.testing.assert_array_equal(data["decision"], p >= meta["threshold"])
                    batches = np.array_split(np.arange(len(sel)), int(np.ceil(len(sel) / 512)))
                    logits = data["logit"].reshape(-1, 5)
                    subtype = data["subtype"].reshape(-1, 5, 4)
                    for probe in (batches[0], batches[len(batches)//2], batches[-1]):
                        blocks = x[sel[probe, None] + np.arange(1-meta["context"], 1)]
                        inputs = (torch.as_tensor(blocks, device=device) - torch.as_tensor(center, device=device))
                        inputs = torch.asinh(inputs / torch.as_tensor(scale, device=device))
                        with torch.inference_mode():
                            scores = model(inputs).cpu().numpy()
                        max_difference = max(max_difference, float(abs(scores[:, :, 0] - logits[probe]).max()))
                        np.testing.assert_allclose(scores[:, :, 0], logits[probe], atol=tolerance)
                        np.testing.assert_allclose(softmax(scores[:, :, 1:], axis=2), subtype[probe], atol=tolerance)
                    aps = [average_precision_score(y[target][ids[target] == i] > 0, p[ids[target] == i])
                           for i in members]
                    if split == "test":
                        patient_aps.extend(aps)
                    else:
                        np.testing.assert_allclose(np.mean(aps), meta["val_patient_ap"], atol=1e-7)
            verified += 1
        np.testing.assert_allclose(np.mean(patient_aps), comparison.loc[name, "patient_ap"], atol=1e-7)
    print(f"Verified {verified} checkpoints, {2*verified} prediction sets, {len(folds)} patients, "
          f"{len(ends)*5:,} held-out seconds per configuration.")
    if source:
        print("Source SHA-256 matches the preparation audit.")
    print(f"Maximum checkpoint replay logit difference on {device.type}: {max_difference:.6g}")
    name = json.loads((RESULT / "selection.json").read_text())["winner"]
    bundle = Path("artifacts/ppg/exports/osasud_v1") / name
    runtime = PPGPredictor(bundle)
    refit = RespiratoryNet(name).eval()
    refit.load_state_dict(torch.load(bundle / "weights.pt", weights_only=True, map_location="cpu"))
    probes = np.random.default_rng(501).choice(ends, 128, replace=False)
    blocks = x[probes[:, None] + np.arange(1-runtime.meta["context"], 1)]
    values = np.arcsinh((blocks-runtime.center)/runtime.scale)
    with torch.inference_mode():
        expected = refit(torch.from_numpy(values)).numpy()
    exported = np.concatenate([runtime.session.run(None, {"views": v[None]})[0] for v in values])
    np.testing.assert_allclose(exported, expected, atol=3e-4, rtol=2e-4)
    print(f"Exported {name} agrees with refit weights on {len(values)} blocks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Optionally verify the original release checksum")
    verify(parser.parse_args().source)
