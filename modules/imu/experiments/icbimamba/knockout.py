"""Knock-out ablations of the trained IC-BiMamba ensemble (no retraining).

Each of the 25 refit members (5 folds x 5 seeds) is loaded and one component is switched off
at inference time. Because the refit members were trained on the validation group, the
threshold for every variant -- including the unmodified "full" reference -- is chosen on that
member's own (in-sample) validation scan, so compare variants with knockout_full, not with
the main run. Writes artifacts/imu/runs/icbimamba_v1/knockout_<variant>/ for evaluate.py.

    .venv/Scripts/python modules/imu/experiments/icbimamba/knockout.py --threads 6
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run as R
from wearable_imu.training import protocol as p
from wearable_imu.models.icbimamba import ICBiMamba

VARIANTS = {
    "full": "Unmodified model (reference)",
    "no_raw": "Raw 200 Hz patch input removed",
    "no_cond": "Impact conditioning (FiLM + dt) removed",
    "unidir": "Backward scan removed (forward only)",
    "no_local": "Multiscale local convolutions removed",
}


def knock(net, variant):
    if variant == "no_raw": net.raw = None
    if variant == "no_cond":
        for b in net.blocks: b.conditioned = False
    if variant == "unidir":
        for b in net.blocks: b.bidirectional = False
    if variant == "no_local":
        net.mix.weight.data.zero_(); net.mix.bias.data.zero_()
    return net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(); torch.set_num_threads(args.threads)
    src = R.OUT / "main"; cfg = json.loads((src / "config.json").read_text())
    c = cfg["cfg"]
    d = p.make_data(R.Experiment("icbimamba", "icbimamba", seconds=4, overlap=.75, scaler="standard"))
    raw = torch.as_tensor(np.asarray(R.raw_tokens(d)))
    outs = {v: R.OUT / f"knockout_{v}" for v in VARIANTS}
    for v, o in outs.items():
        o.mkdir(parents=True, exist_ok=True)
        (o / "config.json").write_text(json.dumps(dict(cfg, knockout=v, description=VARIANTS[v]), indent=2))
    for outer in range(5):
        tr, val, refit, test = p.partitions(d["folds"], outer)
        fin = R.Data(d, raw, refit, "cpu")
        vs = d["starts"][np.isin(d["group"], val)]; ts = d["starts"][np.isin(d["group"], test)]
        for seed in cfg["seeds"]:
            t0 = time.perf_counter()
            state = torch.load(src / f"fold{outer}_seed{seed}.pt", map_location="cpu")
            ref = np.load(src / f"fold{outer}_seed{seed}.npz")
            for v, o in outs.items():
                f = o / f"fold{outer}_seed{seed}.npz"
                if f.is_file(): continue
                net = ICBiMamba(d["names"], width=c["width"], depth=c["depth"], raw=c["raw"],
                                conditioned=c["conditioned"], bidirectional=c["bidirectional"])
                net.load_state_dict(state); knock(net, v)
                test_logits = R.scan(net, fin, ts, d["width"]).astype("f")
                if v == "full":  # sanity: must reproduce the saved GPU test logits
                    err = float(np.abs(test_logits - ref["test"]).max())
                    assert err < 1e-2, f"full model does not reproduce saved logits (max diff {err})"
                np.savez(f, val=R.scan(net, fin, vs, d["width"]).astype("f"), test=test_logits,
                         epoch=ref["epoch"], parameters=ref["parameters"])
            print(f"fold {outer} seed {seed}: {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
