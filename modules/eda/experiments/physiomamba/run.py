"""PhysioMamba WESAD leave-one-subject-out study (same windows, subjects and inner splits as wesad_loso_v1).

Per outer participant: 12 subjects train, 2 fixed inner subjects (+1, +8 in the roster) choose the
epoch for a K-seed ensemble trained in lockstep and fit a scalar temperature; a fresh ensemble is
refit on all 14 for that epoch count and predicts the held-out participant once.

  python modules/eda/experiments/physiomamba/run.py train --models physiomamba
  python modules/eda/experiments/physiomamba/run.py summary
"""
from __future__ import annotations
from pathlib import Path
import sys, json, time, argparse
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "modules/eda/src"))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from wearable_eda.training.neural_suite import (load_windows, prepare, pick_subjects, macro_subject,
                                                temperature, pred_frame, dumps, aggregate)
from wearable_eda.training.neural_experiments import catalogue, arguments, selected_channels
from wearable_eda.models.physiomamba import PhysioMamba, VARIANTS
from wearable_eda.evaluation.loso import load_runs, summarize

CACHE = ROOT / "data/eda/processed/wesad_v1"
BASE = ROOT / "artifacts/eda/runs/wesad_loso_v1"
OUT = ROOT / "artifacts/eda/runs/wesad_physiomamba_v1"
SEEDS, MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 5, 40, 10, 64, 1e-3, 1e-2


def build(variant, names, n_features):
    return PhysioMamba(names, n_features, 2, **VARIANTS[variant])


def logits_of(models, pack, idx):
    out = []
    with torch.inference_mode():
        for m in models:
            m.eval(); out.append(torch.cat([m(pack["x"][p], pack["f"][p]) for p in idx.split(256)]))
    return torch.stack(out).mean(0).cpu().numpy()  # logit ensemble


def train_ensemble(variant, names, nf, pack, seed0, epochs, validation=None):
    device = pack["x"].device
    models, opts, gens = [], [], []
    for k in range(SEEDS):
        torch.manual_seed(seed0 + 1000 * k)
        m = build(variant, names, nf).to(device); models.append(m)
        opts.append(torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD))
        gens.append(torch.Generator(device=device).manual_seed(seed0 + 1000 * k))
    train = pack["train"]
    counts = torch.bincount(pack["y"][train], minlength=2)
    weight = len(train) / (2 * counts.float())
    best, best_epoch, best_logits, history = (-1., -np.inf), 0, None, []
    for epoch in range(1, epochs + 1):
        total = 0.
        for m, opt, g in zip(models, opts, gens):
            m.train()
            for idx in train[torch.randperm(len(train), device=device, generator=g)].split(BATCH):
                x = pack["x"][idx]; f = pack["f"][idx]
                x = x + .05 * torch.randn(x.shape, device=device, generator=g)
                f = f + .05 * torch.randn(f.shape, device=device, generator=g)
                loss = F.cross_entropy(m(x, f), pack["y"][idx], weight=weight, label_smoothing=.05)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.); opt.step()
                total += float(loss.detach()) * len(idx)
        row = dict(epoch=epoch, loss=total / (len(train) * SEEDS))
        if validation is not None:
            idx, y, groups = validation
            logits = logits_of(models, pack, idx)
            p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), 1).numpy()
            metric = macro_subject(p, y, groups); nll = float(-np.log(p[np.arange(len(y)), y].clip(1e-9)).mean())
            row.update(val_macro_f1=metric, val_nll=nll)
            if (metric, -nll) > best: best, best_epoch, best_logits = (metric, -nll), epoch, logits
            if epoch - best_epoch >= PATIENCE: history.append(row); break
        history.append(row)
    return models, history, best_epoch if validation is not None else epochs, best_logits, best[0]


def run_model(variant, calibration, device):
    spec = dict(catalogue()["pg_fullwrist_" + calibration])
    spec.update(id=f"{variant}_{calibration}", model=variant, role="primary" if variant == "physiomamba" else "ablation",
                category="hybrid")
    d = load_windows(CACHE, arguments(spec))
    names = selected_channels(d["channels"], spec)
    raw = d["X"]
    folder = OUT / "neural" / spec["id"]; folder.mkdir(parents=True, exist_ok=True)
    dumps(folder / "meta.json", dict(**spec, seconds=60, stride=15, purity=.8, channels=names, features=list(d["features"]),
          calibrator="inner scalar temperature on K-seed logit ensemble", seeds=SEEDS, max_epochs=MAX_EPOCHS,
          patience=PATIENCE, batch_size=BATCH, lr=LR, weight_decay=WD, device=str(device), torch_version=torch.__version__))
    subjects = np.unique(d["groups"])
    for outer in subjects:
        path = folder / f"S{int(outer):02d}"
        if (path / "fold.json").exists(): continue
        tic = time.perf_counter()
        a, v, r = pick_subjects(d["groups"], outer)
        dev = prepare(d, raw, a, names, device); refit = prepare(d, raw, r, names, device)
        val = np.flatnonzero(np.isin(d["groups"], v)); test = np.flatnonzero(d["groups"] == outer)
        seed = 971 + int(outer)
        _, hist, ep, vlog, score = train_ensemble(variant, names, len(d["features"]), dev, seed, MAX_EPOCHS,
                                                  (torch.as_tensor(val, device=device), d["y"][val], d["groups"][val]))
        temp = temperature(vlog, d["y"][val])
        path.mkdir(parents=True, exist_ok=True)
        pred_frame(d, val, vlog, temp).to_csv(path / "inner_predictions.csv", index=False)
        models, fhist, *_ = train_ensemble(variant, names, len(d["features"]), refit, seed, ep)
        tlog = logits_of(models, refit, torch.as_tensor(test, device=device))
        pred_frame(d, test, tlog, temp).to_csv(path / "predictions.csv", index=False)
        torch.save(dict(variant=variant, channels=names, features=list(d["features"]), temperature=temp,
                        scaler={k: torch.from_numpy(np.asarray(x).copy()) for k, x in refit["scaler"].items()},
                        weights=[{k: t.detach().cpu() for k, t in m.state_dict().items()} for m in models]), path / "model.pt")
        pd.DataFrame([dict(stage=s, **h) for s, rows in (("inner", hist), ("refit", fhist)) for h in rows]).to_csv(path / "history.csv", index=False)
        rec = dict(subject=int(outer), inner_score=float(score), epochs=ep, temperature=temp, seed=seed,
                   trainable_parameters=sum(p.numel() for p in models[0].parameters()),
                   ensemble_members=SEEDS, artifact_bytes=(path / "model.pt").stat().st_size,
                   inner_train_subjects=list(map(int, a)), inner_validation_subjects=list(map(int, v)),
                   refit_subjects=list(map(int, r)), wall_seconds=round(time.perf_counter() - tic, 2))
        dumps(path / "fold.json", rec)
        print(json.dumps({"model": spec["id"], "subject": int(outer), "inner_f1": round(score, 4), "epochs": ep,
                          "seconds": rec["wall_seconds"]}), flush=True)
        del dev, refit, models; torch.cuda.empty_cache()
    aggregate(folder, subjects)


def summary():
    rows = []
    for root in (OUT, BASE):
        for name, run in load_runs(root).items():
            if run.classes != [1, 2]: continue
            row, _ = summarize(run); rows.append(row)
    df = pd.DataFrame(rows)
    keys = ["experiment", "calibration", "modalities", "accuracy", "macro_f1", "macro_f1_sd", "balanced_accuracy",
            "f1", "precision", "sensitivity", "specificity", "auroc", "auprc", "inner_macro_f1"]
    df = df[keys].sort_values("macro_f1", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "comparison_with_wesad_loso_v1.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for cal in ("zero", "rest"):
            print(f"\n== {cal} calibration, top 12 by participant-mean macro F1 ==")
            print(df[df.calibration == cal].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["train", "summary"])
    p.add_argument("--models", nargs="+", default=["physiomamba"])
    p.add_argument("--calibrations", nargs="+", default=["zero", "rest"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.phase == "train":
        for c in a.calibrations:
            for m in a.models: run_model(m, c, torch.device(a.device))
    else: summary()
