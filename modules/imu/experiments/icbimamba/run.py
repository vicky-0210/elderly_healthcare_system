"""IC-BiMamba on SisFall: same 5 participant-disjoint outer folds and operating-point rule as sisfall_v1.

Per outer fold k: group (k+1)%5 is validation (epoch, threshold, calibration), the other three train;
each seed is then refit on all four development groups for its selected epoch and scores group k once.
Seeds are ensembled by averaging window logits; the trial decision is the max over every window.

Run from the repository root:
    .venv/Scripts/python modules/imu/experiments/icbimamba/run.py --tag main
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wearable_imu.training.catalogue import Experiment
from wearable_imu.training import protocol as p
from wearable_imu.datasets.sisfall import CACHE, COUNTS_TO_PHYSICAL, load_cache
from wearable_imu.models.icbimamba import ICBiMamba, PATCH

OUT = Path("artifacts/imu/runs/icbimamba_v1")
SEEDS = (4907, 6203, 7919, 1361, 2718)


def raw_tokens(d):
    path = Path(CACHE) / "raw_patch_f16.npy"
    total = int(d["lengths"].sum())
    if path.is_file():
        arr = np.load(path, mmap_mode="r")
        if arr.shape == (total, PATCH * 6): return arr
    manifest, counts = load_cache(CACHE)
    out = np.empty((total, PATCH * 6), dtype=np.float16)
    for row, start, length in zip(manifest.itertuples(), d["offs"], d["lengths"]):
        seg = counts[row.offset:row.offset + length * PATCH].astype("f") * COUNTS_TO_PHYSICAL
        out[start:start + length] = seg.reshape(length, PATCH * 6)
    np.save(path, out)
    return out


class Data:
    def __init__(self, d, raw, ids, device):
        center, scale = p.fit_scaler(d["x"], ids, d["offs"], d["lengths"], "standard")
        x = torch.as_tensor(np.asarray(d["x"]), device=device)
        self.x = torch.asinh((x - torch.tensor(center, device=device)) / torch.tensor(scale, device=device))
        self.raw = raw  # shared GPU tensor (T,60) float16


def windows(data, starts, width, steps, augment=False):
    index = starts[:, None] + steps
    x = data.x[index].transpose(1, 2)
    r = data.raw[index].float()
    if augment:
        gain = 1 + .04 * torch.randn(len(starts), 1, 1, 6, device=r.device)
        r = (r.reshape(len(starts), width, PATCH, 6) * gain).reshape(len(starts), width, -1)
        x = x + .03 * torch.randn_like(x)
    return x, r


def scan(net, data, starts, width):
    net.eval(); outs = []
    steps = torch.arange(width, device=data.x.device)
    with torch.inference_mode():
        for block in torch.split(torch.as_tensor(starts, device=data.x.device), 512):
            outs.append(net(*windows(data, block, width, steps)).float().cpu().numpy())
    return np.concatenate(outs)


def train(d, data, ids, seed, cfg, val=None, epochs=None):
    device = data.x.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = ICBiMamba(d["names"], width=cfg["width"], depth=cfg["depth"], raw=cfg["raw"],
                    conditioned=cfg["conditioned"], bidirectional=cfg["bidirectional"]).to(device)
    total_epochs = cfg["epochs"]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    per_epoch = int(np.ceil(len(ids) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=total_epochs * per_epoch, pct_start=.15)
    width = d["width"]; steps = torch.arange(width, device=device)
    sizes = torch.as_tensor(np.bincount(d["group"]), device=device)
    first = torch.as_tensor(d["first"], device=device)
    all_starts = torch.as_tensor(d["starts"], device=device)
    y = torch.as_tensor(d["y"], dtype=torch.float32, device=device)
    pos = torch.tensor((d["y"][ids] == 0).sum() / (d["y"][ids] == 1).sum(), device=device)
    if val is not None: val_starts = d["starts"][np.isin(d["group"], val)]
    best, history = None, []
    stop_at = epochs or total_epochs
    for epoch in range(1, stop_at + 1):
        net.train(); total = 0.
        for batch in np.array_split(rng.permutation(ids), per_epoch):
            idx = torch.as_tensor(batch, device=device)
            k = cfg["bag"]
            pick = (torch.rand(len(idx), k, device=device) * sizes[idx, None]).long()
            loc = all_starts[first[idx, None] + pick].reshape(-1)
            logits = net(*windows(data, loc, width, steps, augment=cfg["augment"])).reshape(len(idx), k)
            loss = F.binary_cross_entropy_with_logits(logits.amax(1), y[idx], pos_weight=pos)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step(); sched.step()
            total += float(loss.detach()) * len(batch)
        rec = dict(epoch=epoch, loss=total / len(ids))
        if not np.isfinite(rec["loss"]): raise FloatingPointError("non-finite loss")
        if val is not None:
            vl = scan(net, data, val_starts, width)
            bag = p.trial_max(d, val, vl)
            score, thr = p.operating_point(d["y"][val], bag, d["subjects"][val])
            rec["validation"] = score
            # ties go to the later epoch (more annealed)
            if best is None or score >= best[0] - 1e-9: best = (score, thr, epoch, vl)
            if epoch - best[2] >= cfg["patience"]: history.append(rec); break
        history.append(rec)
    return net, history, best


CFG = dict(width=48, depth=3, raw=True, conditioned=True, bidirectional=True, lr=2e-3, wd=.02, batch=48, bag=8,
           epochs=30, patience=8, augment=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--inner-only", action="store_true", help="development: no refit, no test scoring")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. raw=0 depth=2")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    args = ap.parse_args()
    if args.threads: torch.set_num_threads(args.threads)
    cfg = dict(CFG)
    for kv in args.set:
        k, v = kv.split("="); cfg[k] = type(CFG[k])(float(v)) if not isinstance(CFG[k], bool) else bool(int(v))
    spec = Experiment("icbimamba", "icbimamba", seconds=4, overlap=.75, scaler="standard")
    d = p.make_data(spec)
    device = args.device
    raw = torch.as_tensor(np.asarray(raw_tokens(d)), device=device)
    out = OUT / args.tag; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg=cfg, seeds=args.seeds, seconds=4, overlap=.75), indent=2))
    for outer in args.folds:
        tr, val, refit, test = p.partitions(d["folds"], outer)
        dev = Data(d, raw, tr, device); fin = None
        for seed in args.seeds:
            f = out / f"fold{outer}_seed{seed}.npz"
            if f.is_file() and (args.inner_only or "test" in np.load(f).files): continue
            t0 = time.perf_counter()
            _, hist, (score, thr, epoch, vl) = train(d, dev, tr, seed + outer, cfg, val=val)
            rec = dict(val=vl.astype("f"), epoch=epoch, score=score, threshold=thr, history=json.dumps(hist))
            if not args.inner_only:
                fin = fin or Data(d, raw, refit, device)
                net, _, _ = train(d, fin, refit, seed + outer, cfg, epochs=epoch)
                rec["test"] = scan(net, fin, d["starts"][np.isin(d["group"], test)], d["width"]).astype("f")
                rec["parameters"] = sum(v.numel() for v in net.parameters())
                torch.save(net.state_dict(), out / f"fold{outer}_seed{seed}.pt")
            np.savez(f, **rec)
            print(f"fold {outer} seed {seed}: val BA {score:.4f} @ epoch {epoch} ({time.perf_counter()-t0:.0f}s)", flush=True)
        del dev, fin; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
