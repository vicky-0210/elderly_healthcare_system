"""Inference-time modality knock-out of the trained PhysioMamba ensembles (no retraining).

For every held-out participant the saved 5-member refit ensemble is re-scored with one modality group
(cardiac/BVP or EDA) switched off exactly as modality dropout does during training (stream zeroed
after its embedding). Reference and knock-outs use the stored temperature and argmax decision.

    .venv/Scripts/python modules/eda/experiments/physiomamba/knockout.py --calibration zero
Writes research/results/physiomamba_knockout/<calibration>.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "modules/eda/src"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import run as R
from wearable_eda.training.neural_suite import load_windows, prepare, pick_subjects, pred_frame
from wearable_eda.training.neural_experiments import catalogue, arguments, selected_channels

OUT = ROOT / "research/results/physiomamba_knockout"
KNOCK = {"full": None, "no_cardiac": "cardiac", "no_eda": "eda", "no_motion": "motion"}


def knocked(model, group):
    if group is None or group not in model.groups: return model
    i = list(model.groups).index(group)
    emb = model.embed[group]

    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(emb(x))
    model.embed[group] = Zero(); model.modality.data[i].zero_()
    return model


def subject_metrics(y, p):
    pred = p.argmax(1); tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f1s = 2 * tp / max(2 * tp + fp + fn, 1); f1b = 2 * tn / max(2 * tn + fn + fp, 1)
    return dict(accuracy=(tp + tn) / len(y), macro_f1=(f1s + f1b) / 2, sensitivity=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), auroc=roc_auc_score(y, p[:, 1]) if len(set(y)) == 2 else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calibration", default="zero"); args = ap.parse_args()
    torch.set_num_threads(4)
    spec = dict(catalogue()["pg_fullwrist_" + args.calibration]); spec.update(model="physiomamba")
    d = load_windows(R.CACHE, arguments(spec)); names = selected_channels(d["channels"], spec)
    folder = R.OUT / "neural" / f"physiomamba_{args.calibration}"
    per = {k: {} for k in KNOCK}
    for outer in np.unique(d["groups"]):
        _, _, r = pick_subjects(d["groups"], outer)
        pack = prepare(d, d["X"], r, names, "cpu")
        test = np.flatnonzero(d["groups"] == outer)
        saved = torch.load(folder / f"S{int(outer):02d}/model.pt", map_location="cpu", weights_only=False)
        for k, g in KNOCK.items():
            models = []
            for w in saved["weights"]:
                m = R.build("physiomamba", names, len(d["features"])); m.load_state_dict(w); models.append(knocked(m, g))
            logits = R.logits_of(models, pack, torch.as_tensor(test))
            fr = pred_frame(d, test, logits, saved["temperature"])
            per[k][int(outer)] = subject_metrics(fr.y_true.to_numpy(), fr[["p0", "p1"]].to_numpy())
        print(outer, {k: round(per[k][int(outer)]["macro_f1"], 3) for k in KNOCK}, flush=True)
    out = {}
    rng = np.random.default_rng(0)
    for k in KNOCK:
        subs = sorted(per[k]); m = {c: float(np.nanmean([per[k][s][c] for s in subs])) for c in per[k][subs[0]]}
        if k != "full":
            dlt = np.array([per["full"][s]["macro_f1"] - per[k][s]["macro_f1"] for s in subs])
            bs = dlt[rng.integers(0, len(dlt), (2000, len(dlt)))].mean(1)
            m["delta_macro_f1"] = [float(dlt.mean()), *map(float, np.quantile(bs, [.025, .975]))]
            m["better_worse"] = [int((dlt > 0).sum()), int((dlt < 0).sum())]
        out[k] = dict(mean=m, per_subject=per[k])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.calibration}.json").write_text(json.dumps(out, indent=1, default=float))
    for k in KNOCK: print(k, {c: round(v, 4) if isinstance(v, float) else v for c, v in out[k]["mean"].items()})


if __name__ == "__main__":
    main()
