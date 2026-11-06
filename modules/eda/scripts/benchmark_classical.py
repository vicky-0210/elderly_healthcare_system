"""Benchmark one held-out WESAD checkpoint per zero-calibration classical family."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE/'src'))
from wearable_eda.training import classical
from wearable_eda.training.classical_run import feature_data, representation, source_state


def worker():
    request = json.load(sys.stdin)
    example = np.asarray(request['x'], dtype=np.float64)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    bundle = classical.load_bundle(request['file'])
    arguments = bundle, example, request['features']
    with threadpool_limits(1):
        for _ in range(20):
            classical.predict_windows(*arguments)
        rss_warm = process.memory_info().rss
        wall, cpu = [], []
        for _ in range(300):
            start = perf_counter_ns()
            cpu_start = process_time_ns()
            classical.predict_windows(*arguments)
            cpu.append((process_time_ns() - cpu_start) / 1e6)
            wall.append((perf_counter_ns() - start) / 1e6)
        process_memory = process.memory_info()
    return dict(artifact_bytes=Path(request['file']).stat().st_size,
                host=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                baseline_rss_bytes=rss_before, warm_process_rss_bytes=rss_warm,
                rss_delta_bytes=rss_warm-rss_before, peak_process_rss_bytes=getattr(process_memory, 'peak_wset', None),
                n_repeats=300, warmups=20, cpu_threads=1, batch_size=1,
                wall_p50_ms=float(np.median(wall)), wall_p95_ms=float(np.percentile(wall, 95)),
                process_cpu_mean_ms=float(np.mean(cpu)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/eda/processed/wesad_v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/eda/runs/wesad_loso_v1/classical'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker()))
        return
    eligible = {}
    for path in sorted(args.output.glob('*/meta.json')):
        meta = json.loads(path.read_text())
        if meta['role'] == 'candidate' and meta['calibration'] == 'zero':
            score = pd.read_csv(path.with_name('folds.csv')).inner_score.mean()
            if meta['model'] not in eligible or score > eligible[meta['model']][0]:
                eligible[meta['model']] = score, path.parent, meta
    if set(eligible) != set(classical.GRIDS):
        raise ValueError('Finish all classical candidate runs before benchmarking')
    rows = []
    state = source_state(args.cache)
    for name, (score, folder, meta) in eligible.items():
        data = feature_data(args.cache, representation(meta), state)
        x = data['F'][data['groups'] == 2][0]
        checkpoint = folder/'subjects'/'S2'/'model.joblib'
        packet = dict(file=str(checkpoint.resolve()), x=x.tolist(), features=data['features'])
        env = os.environ | {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS','MKL_NUM_THREADS')}
        reply = subprocess.run([sys.executable, __file__, '--worker'], input=json.dumps(packet),
                               text=True, check=True, capture_output=True, env=env)
        rows.append(dict(experiment=meta['name'], model=name, subject=2, inner_macro_f1=score,
                         **json.loads(reply.stdout)))
    result = pd.DataFrame(rows)
    out = args.output/'cpu_benchmark.csv'
    result.to_csv(out, index=False)
    print(result[['experiment','artifact_bytes','warm_process_rss_bytes','wall_p50_ms','wall_p95_ms']].to_string(index=False))
    print(out)


if __name__ == '__main__':
    main()
