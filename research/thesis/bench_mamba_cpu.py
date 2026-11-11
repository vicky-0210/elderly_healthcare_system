"""CPU latency of the three proposed Mamba-2 models (one window, batch 1, PyTorch eager, single member).

    .venv/Scripts/python research/thesis/bench_mamba_cpu.py
Writes research/thesis/figures/v4/cpu_latency.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for m in ("ecg", "imu", "eda"): sys.path.insert(0, str(ROOT / f"modules/{m}/src"))
from wearable_ecg.models import vcmamba
from wearable_imu.models.icbimamba import ICBiMamba, CONDITION
from wearable_eda.models.physiomamba import PhysioMamba, GROUPS


def bench(net, args, n=200, warm=20):
    net.eval(); times = []
    with torch.inference_mode():
        for i in range(warm + n):
            t0 = time.perf_counter(); net(*args); dt = time.perf_counter() - t0
            if i >= warm: times.append(dt * 1000)
    t = np.array(times)
    return dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)),
                params=int(sum(p.numel() for p in net.parameters())))


torch.manual_seed(0)
imu_names = list(CONDITION) + [f"f{i}" for i in range(24 - len(CONDITION))]
eda_channels = [c for g in GROUPS.values() for c in g]
models = {
    "VC-BiMamba (ECG, 10 s)": (vcmamba.make_novel("vc_bimamba"), (torch.randn(1, 3600) * .5, torch.randn(1, 16)), 2000),
    "IC-BiMamba (IMU, 4 s)": (ICBiMamba(imu_names), (torch.randn(1, 24, 80), torch.randn(1, 80, 60)), 1000),
    "PhysioMamba (EDA, 60 s)": (PhysioMamba(eda_channels, 39, 2), (torch.randn(1, len(eda_channels), 240), torch.randn(1, 39)), 15000),
}
out = {}
for threads in (1, 4):
    torch.set_num_threads(threads)
    for name, (net, args, hop) in models.items():
        r = bench(net, args); r["hop_ms"] = hop; r["budget_pct_p95_5members"] = 5 * r["p95"] / hop * 100
        out[f"{name}|{threads}"] = r
        print(f"{threads} thr  {name:26s} p50 {r['p50']:.2f} p95 {r['p95']:.2f} ms  params {r['params']:,}  5-member budget {r['budget_pct_p95_5members']:.2f}%")
cpu = "Intel Core i5-11400H (laptop), PyTorch " + torch.__version__
(ROOT / "research/thesis/figures/v4/cpu_latency.json").write_text(json.dumps(dict(cpu=cpu, results=out), indent=1))
