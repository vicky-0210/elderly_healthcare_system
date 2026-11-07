"""Matched follow-up on the frozen OSASUD v1 folds and the original scored seconds."""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.nn import functional as F

from wearable_ppg.datasets.osasud import CACHE, RESULT, RUN
from wearable_ppg.datasets.windows import load_views
from wearable_ppg.evaluation.metrics import choose_threshold, patient_ap
from wearable_ppg.models.dual_rate import FRACTION, make_model
from wearable_ppg.training.study import EPOCHS, SEED, calibration, norm_fit, sample_train, seconds


RUN_DIR = RUN.parent / "osasud_binary_dual_rate_v1"
REPORT_DIR = RESULT.parent / "osasud_binary_dual_rate_v1"


def run_starts(timestamps, valid):
    """Earliest permitted history index at each row, across packet losses and clock jumps."""
    valid = np.asarray(valid, dtype=bool)
    boundaries = np.r_[True, (~valid[:-1]) | (~valid[1:]) | (np.diff(timestamps) != 1)]
    return np.maximum.accumulate(np.where(boundaries, np.arange(len(valid)), 0))


def load_data():
    frames, y, t, ids, ends = load_views()
    starts = []
    offset = 0
    for patient in pd.read_csv(RESULT / "folds.csv").patient:
        with np.load(CACHE / "views" / f"p{patient:02}.npz") as item:
            n = len(item["valid"])
            np.testing.assert_array_equal(item["t"], t[offset:offset+n])
            assert np.all(ids[offset:offset+n] == patient)
            starts.append(run_starts(item["t"], item["valid"]) + offset)
            offset += n
    starts = np.concatenate(starts)
    assert np.all(ends - starts[ends] >= 29)
    return frames, y, t, ids, ends, starts


def trailing_targets(y, ends):
    positions = seconds(ends)
    hard = y[positions] > 0
    fraction = (y[positions[:, :, None] + np.arange(1-FRACTION, 1)] > 0).mean(-1)
    return hard.astype("f"), fraction.astype("f")


def window_batch(x, starts, ends, width):
    end = torch.as_tensor(ends, dtype=torch.long, device=x.device)
    index = end[:, None] + torch.arange(1-width, 1, device=x.device)
    mask = index >= starts[end, None]
    values = x[index.clamp_min(0)].masked_fill(~mask[:, :, None], 0)
    return torch.cat((values, mask[:, :, None].to(x.dtype)), dim=-1)


def infer(net, x, starts, ends):
    net.eval()
    output = []
    with torch.inference_mode():
        for batch in np.array_split(ends, int(np.ceil(len(ends)/512))):
            logits = net(window_batch(x, starts, batch, net.context))
            output.append(logits[:, :, 0].cpu().numpy().ravel())
    return np.concatenate(output)


def fit(name, x, starts, y, ids, train, valid, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_model(name).to(x.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.005)
    history = []
    best = -np.inf
    val_idx = seconds(valid).ravel() if valid is not None else None
    for epoch in range(1, epochs+1):
        net.train()
        losses = []
        sampled = sample_train(train, ids, y, rng)
        for batch in np.array_split(sampled, int(np.ceil(len(sampled)/192))):
            hard, fraction = trailing_targets(y, batch)
            logits = net(window_batch(x, starts, batch, net.context))
            target = fraction if name == "dual_fraction" else hard
            loss = F.binary_cross_entropy_with_logits(logits[:, :, 0], torch.as_tensor(target, device=x.device))
            if name == "dual_aux":
                loss = loss + .35*F.binary_cross_entropy_with_logits(
                    logits[:, :, 1], torch.as_tensor(fraction, device=x.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            optimizer.step()
            losses.append(loss.detach())
        entry = dict(epoch=epoch, loss=torch.stack(losses).mean().item())
        history.append(entry)
        if valid is None:
            continue
        logits = infer(net, x, starts, valid)
        value = patient_ap(y[val_idx], logits, ids[val_idx])
        entry["val_patient_ap"] = value
        if value > best + .0001:
            best = value
            best_epoch = epoch
            state = copy.deepcopy(net.state_dict())
        if epoch - best_epoch >= 3:
            break
    if valid is not None:
        net.load_state_dict(state)
    return net, history, best_epoch if valid is not None else epochs


def scores(net, x, starts, ends, y, t, ids):
    idx = seconds(ends).ravel()
    return dict(patient=ids[idx], t=t[idx], y=y[idx], arrival=np.repeat(t[ends]+1, 5),
                logit=infer(net, x, starts, ends))


def run(names, device):
    torch.set_num_threads(2)
    frames, y, t, ids, ends, starts = load_data()
    starts = torch.as_tensor(starts, device=device)
    manifest = pd.read_csv(RESULT / "folds.csv")
    for k in range(5):
        test_ids = manifest.loc[manifest.fold == k, "patient"].tolist()
        val_ids = manifest.loc[manifest.fold == (k+1)%5, "patient"].tolist()
        train_ids = sorted(set(manifest.patient)-set(test_ids)-set(val_ids))
        train, valid, test = (ends[np.isin(ids[ends], group)] for group in (train_ids, val_ids, test_ids))
        center, scale = norm_fit(frames, train)
        x = torch.asinh((torch.as_tensor(frames, device=device)-torch.tensor(center, device=device))
                        /torch.tensor(scale, device=device))
        for name in names:
            path = RUN_DIR / name / f"fold{k}"
            path.mkdir(parents=True, exist_ok=True)
            if (path / "done.json").exists():
                continue
            started = time.perf_counter()
            net, history, epoch = fit(name, x, starts, y, ids, train, valid, SEED+k)
            val = scores(net, x, starts, valid, y, t, ids)
            a, b = calibration(val["y"], val["logit"])
            val_p = expit(a*val["logit"]+b)
            threshold = choose_threshold(val["y"], val_p, val["t"], val["patient"], val["arrival"])
            held = scores(net, x, starts, test, y, t, ids)
            for label, rows in (("validation", val), ("test", held)):
                p = expit(a*rows["logit"]+b)
                np.savez_compressed(path / f"{label}.npz", **rows, p=p, decision=p>=threshold)
            torch.save({key: value.cpu() for key, value in net.state_dict().items()}, path / "model.pt")
            np.savez(path / "scaler.npz", center=center, scale=scale)
            meta = dict(model=name, fold=k, train=train_ids, val=val_ids, test=test_ids, seed=SEED+k,
                        context=net.context, threshold=threshold, calibration=[a,b], epoch=epoch,
                        parameters=sum(p.numel() for p in net.parameters()), history=history,
                        val_patient_ap=patient_ap(val["y"], val_p, val["patient"]),
                        wall_seconds=time.perf_counter()-started)
            marker = path / "done.tmp"
            marker.write_text(json.dumps(meta, indent=2))
            marker.replace(path / "done.json")
            print(f"{name} fold {k}: epoch {epoch}, validation AP {meta['val_patient_ap']:.4f}, "
                  f"{meta['wall_seconds']:.0f}s", flush=True)
