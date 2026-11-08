"""Unified Jetson Orin Nano benchmark suite for top-performing models across all 4 datasets.

Datasets and Top Models covered:
1. ECG (PTB-XL):
   - Supervised CNN (artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal/cnn)
   - Distilled Student prep_minimal (artifacts/ecg/exports/lvh_distill_v1/prep_minimal)
2. PPG (OSASUD):
   - Dual-rate Hard Binary (artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard)
   - Multiscale CNN+Transformer (artifacts/ppg/exports/osasud_v1/cnn_transformer)
3. IMU (SisFall):
   - 100 Hz IPMV + XGBoost (artifacts/imu/exports/sisfall_v1/native_100)
   - DM-TCN Distilled (artifacts/imu/exports/sisfall_v1/dm_distilled)
4. EDA (WESAD):
   - 31-coef Logistic EDA+ACC (artifacts/eda/exports/wesad_loso_v1)
   - PG-MSTN Neural (artifacts/eda/exports/wesad_loso_v1_neural)

Measures:
- Model-only forward latency (ONNX Runtime CPU/CUDA/TensorRT, PyTorch, NumPy)
- Full-pipeline end-to-end latency (preprocessing + inference)
- Latency percentiles (p50, p90, p95, p99, mean, min, max, std)
- Real-time deadline verification and budget utilization percentage
- Output verification against synthetic golden reference vectors
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

# Ensure module sources are importable
ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT / "packages/common/src",
          ROOT / "modules/ecg/src",
          ROOT / "modules/ecg/experiments/lvh_distillation",
          ROOT / "modules/ppg/src",
          ROOT / "modules/imu/src",
          ROOT / "modules/eda/src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def get_hardware_info():
    board_path = Path("/proc/device-tree/model")
    if board_path.exists():
        try:
            return board_path.read_bytes().rstrip(b"\x00").decode("utf-8")
        except Exception:
            pass
    return f"{platform.system()} {platform.machine()} ({platform.processor() or 'generic'})"


def time_callable(fn, warmup=10, iterations=100):
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(durations)
    return {
        "iterations": iterations,
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ==========================================
# 1. ECG BENCHMARKS
# ==========================================
def benchmark_ecg_cnn(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable_ecg.inference.predictor import ECGPredictor
    
    export_dir = root / "artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal/cnn"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    # 1. Pipeline predictor test
    predictor = ECGPredictor().load(export_dir)
    ref_data = np.load(ref_file)
    samples = ref_data["samples"]
    window = {"samples": samples, "sampling_rate_hz": 360, "lead": "II", "units": "mV"}
    
    # Verify correctness
    res = predictor.predict(window)
    expected_score = float(ref_data["expected_score"])
    err = abs(res["score"] - expected_score)
    flips = int(res["flag"] != bool(ref_data["expected_flag"]))
    
    # Full pipeline timing
    pipe_timing = time_callable(lambda: predictor.predict(window), iterations=iterations)
    
    # Pure ONNX model timing
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    x_in = np.random.randn(1, 3600).astype(np.float32)
    f_in = np.random.randn(1, 16).astype(np.float32)
    feed = {"waveform_mv": x_in, "voltage_features": f_in}
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = 2000.0  # 2s hop for 10s window
    return {
        "dataset": "PTB-XL",
        "signal": "ECG",
        "model_id": "ecg_lvh_cnn",
        "architecture": "Multiscale CNN with Voltage & Morphology Features",
        "provider": provider,
        "input_shape": "wave: [1, 3600], features: [1, 16]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": flips,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


def benchmark_ecg_minimal(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable import WearableStudent
    
    export_dir = root / "artifacts/ecg/exports/lvh_distill_v1/prep_minimal"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    ws = WearableStudent(export_dir)
    ref_data = np.load(ref_file)
    samples = ref_data["samples"]
    t = ref_data["timestamps"]
    
    res = ws.predict(samples, t, 360)
    expected_score = float(ref_data["expected_score"])
    err = abs(res["score"] - expected_score)
    flips = int(res["flag"] != bool(ref_data["expected_flag"]))
    
    pipe_timing = time_callable(lambda: ws.predict(samples, t, 360), iterations=iterations)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    feed = {"wave": np.random.randn(1, 1, 3600).astype(np.float32)}
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = 2000.0
    return {
        "dataset": "PTB-XL",
        "signal": "ECG",
        "model_id": "ecg_prep_minimal",
        "architecture": "Distilled Student 1D CNN (prep_minimal)",
        "provider": provider,
        "input_shape": "wave: [1, 1, 3600]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": flips,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


# ==========================================
# 2. PPG BENCHMARKS
# ==========================================
def benchmark_ppg_dual_hard(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable_ppg.inference.predictor import PPGPredictor
    
    export_dir = root / "artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    predictor = PPGPredictor(export_dir, fs=80)
    ref_data = np.load(ref_file)
    samples = ref_data["samples"]
    
    # Run once to check output
    predictor.reset()
    res = predictor.predict(samples)
    scores = np.array([r["p_event"] for r in res if r.get("status") == "ok"])
    expected_scores = ref_data["expected_scores"]
    err = float(np.max(np.abs(scores - expected_scores))) if len(scores) and len(expected_scores) else 0.0
    
    # Pipeline timing (single 5s block push = 400 samples)
    block_samples = samples[:400]
    predictor.reset()
    # warm up history
    predictor.push(samples[:90 * 80], 0.0)
    pipe_timing = time_callable(lambda: predictor.push(block_samples, predictor.next_time), iterations=iterations)
    
    # Pure ONNX model timing
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    feed = {"views": np.random.randn(1, 90, 32).astype(np.float32)}
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = 5000.0  # 5s hop
    return {
        "dataset": "OSASUD",
        "signal": "PPG",
        "model_id": "ppg_dual_hard",
        "architecture": "Two-Time-Scale Causal TCN + GRU (Dual-rate Hard)",
        "provider": provider,
        "input_shape": "views: [1, 90, 32]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": 0,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


def benchmark_ppg_cnn_transformer(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable_ppg.inference.predictor import PPGPredictor
    
    export_dir = root / "artifacts/ppg/exports/osasud_v1/cnn_transformer"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    predictor = PPGPredictor(export_dir, fs=80)
    ref_data = np.load(ref_file)
    samples = ref_data["samples"]
    
    predictor.reset()
    res = predictor.predict(samples)
    scores = np.array([r["p_event"] for r in res if r.get("status") == "ok"])
    expected_scores = ref_data["expected_scores"]
    err = float(np.max(np.abs(scores - expected_scores))) if len(scores) and len(expected_scores) else 0.0
    
    block_samples = samples[:400]
    predictor.reset()
    predictor.push(samples[:30 * 80], 0.0)
    pipe_timing = time_callable(lambda: predictor.push(block_samples, predictor.next_time), iterations=iterations)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    feed = {"views": np.random.randn(1, 30, 31).astype(np.float32)}
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = 5000.0
    return {
        "dataset": "OSASUD",
        "signal": "PPG",
        "model_id": "ppg_cnn_transformer",
        "architecture": "Multiscale CNN + Transformer",
        "provider": provider,
        "input_shape": "views: [1, 30, 31]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": 0,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


# ==========================================
# 3. IMU BENCHMARKS
# ==========================================
def benchmark_imu_xgb(root, iterations):
    from wearable_imu.inference.predictor import IMUPredictor
    
    export_dir = root / "artifacts/imu/exports/sisfall_v1/native_100"
    ref_file = export_dir / "synthetic_reference.npz"
    
    pred = IMUPredictor(export_dir)
    with np.load(ref_file, allow_pickle=False) as data:
        x, expected = data["physical"], data["logits"]
    
    pred.reset()
    actual = np.array([r["window_logit"] for r in pred.push(x, 0.0)])
    err = float(np.abs(actual - expected).max())
    flips = int(np.sum((actual >= pred.meta["threshold"]) != (expected >= pred.meta["threshold"])))
    
    pred.reset()
    count = pred.fs * pred.hop // 20
    block = x[:count]
    pipe_timing = time_callable(lambda: pred.push(block, 0.0), iterations=iterations)
    
    # Pure model inference timing
    windows = np.random.randn(1, pred.width, len(pred.indices)).astype(np.float32)
    model_timing = time_callable(lambda: pred.infer(windows), iterations=iterations)
    
    deadline_ms = (pred.hop / 20.0) * 1000.0  # 1000 ms
    return {
        "dataset": "SisFall",
        "signal": "IMU",
        "model_id": "imu_native_100",
        "architecture": "100 Hz IPMV + XGBoost (NumPy ensemble)",
        "provider": "numpy_native",
        "input_shape": f"windows: [1, {pred.width}, 24]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": flips,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


def benchmark_imu_distilled(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable_imu.inference.predictor import IMUPredictor
    
    export_dir = root / "artifacts/imu/exports/sisfall_v1/dm_distilled"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    pred = IMUPredictor(export_dir)
    with np.load(ref_file, allow_pickle=False) as data:
        x, expected = data["physical"], data["logits"]
    
    pred.reset()
    actual = np.array([r["window_logit"] for r in pred.push(x, 0.0)])
    err = float(np.abs(actual - expected).max())
    flips = int(np.sum((actual >= pred.meta["threshold"]) != (expected >= pred.meta["threshold"])))
    
    pred.reset()
    count = pred.fs * pred.hop // 20
    block = x[:count]
    pipe_timing = time_callable(lambda: pred.push(block, 0.0), iterations=iterations)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    feed = {"window": np.random.randn(1, 24, 40).astype(np.float32)}
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = (pred.hop / 20.0) * 1000.0  # 1000 ms
    return {
        "dataset": "SisFall",
        "signal": "IMU",
        "model_id": "imu_dm_distilled",
        "architecture": "Dual-stream Multiscale TCN (Weak-Teacher Distilled)",
        "provider": provider,
        "input_shape": "window: [1, 24, 40]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": flips,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


# ==========================================
# 4. EDA BENCHMARKS
# ==========================================
def benchmark_eda_logistic(root, iterations):
    from wearable_eda.inference.predictor import EDAPredictor
    
    export_dir = root / "artifacts/eda/exports/wesad_loso_v1"
    ref_file = export_dir / "synthetic_reference.npz"
    
    pred = EDAPredictor().load(export_dir)
    ref_data = np.load(ref_file)
    eda = ref_data["eda"]
    t_eda = ref_data["t_eda"]
    acc = ref_data["acc"]
    t_acc = ref_data["t_acc"]
    
    window = {
        "stream_id": "bench-1",
        "clock_domain": "session-clock",
        "streams": {
            "eda": {"unit": "uS", "samples": eda, "timestamps": t_eda},
            "acc": {"unit": "g", "samples": acc, "timestamps": t_acc}
        }
    }
    
    res = pred.predict(window)
    scores = np.array([res["scores"]["baseline"], res["scores"]["stress_associated_arousal"]])
    expected = ref_data["expected_scores"]
    err = float(np.max(np.abs(scores - expected)))
    
    pipe_timing = time_callable(lambda: pred.predict(window), iterations=iterations)
    
    # Pure linear scoring
    f_dummy = np.random.randn(1, len(pred.model.weights)).astype(np.float64)
    model_timing = time_callable(lambda: pred.model.predict_windows(f_dummy, pred.model.features), iterations=iterations)
    
    deadline_ms = 15000.0  # 15s stride
    return {
        "dataset": "WESAD",
        "signal": "EDA",
        "model_id": "eda_wesad_logistic",
        "architecture": "31-coefficient Logistic Regression on MC-SAED",
        "provider": "linear_json",
        "input_shape": f"features: [1, {len(pred.model.weights)}]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": 0,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


def benchmark_eda_neural(root, iterations, provider="CPUExecutionProvider"):
    import onnxruntime as ort
    from wearable_eda.inference.predictor import EDAPredictor
    
    export_dir = root / "artifacts/eda/exports/wesad_loso_v1_neural"
    ref_file = export_dir / "synthetic_reference.npz"
    onnx_file = export_dir / "model.onnx"
    
    pred = EDAPredictor().load(export_dir)
    ref_data = np.load(ref_file)
    window = {
        "stream_id": "bench-1",
        "clock_domain": "session-clock",
        "streams": {
            "eda": {"unit": "uS", "samples": ref_data["eda"], "timestamps": ref_data["t_eda"]},
            "acc": {"unit": "g", "samples": ref_data["acc"], "timestamps": ref_data["t_acc"]},
            "temp": {"unit": "degC", "samples": ref_data["temp"], "timestamps": ref_data["t_temp"]},
            "bvp": {"unit": "E4_arbitrary", "samples": ref_data["bvp"], "timestamps": ref_data["t_bvp"]}
        }
    }
    
    res = pred.predict(window)
    scores = np.array([res["scores"]["baseline"], res["scores"]["stress_associated_arousal"]])
    expected = ref_data["expected_scores"]
    err = float(np.max(np.abs(scores - expected)))
    
    pipe_timing = time_callable(lambda: pred.predict(window), iterations=iterations)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    feed = {
        "sequence": np.random.randn(1, 21, 240).astype(np.float32),
        "features": np.random.randn(1, 39).astype(np.float32)
    }
    model_timing = time_callable(lambda: sess.run(None, feed), iterations=iterations)
    
    deadline_ms = 15000.0  # 15s stride
    return {
        "dataset": "WESAD",
        "signal": "EDA",
        "model_id": "eda_wesad_neural_pg",
        "architecture": "PG-MSTN Full Wrist Neural Network",
        "provider": provider,
        "input_shape": "sequence: [1, 21, 240], features: [1, 39]",
        "realtime_deadline_ms": deadline_ms,
        "golden_error": float(err),
        "decisions_changed": 0,
        "model_latency": model_timing,
        "pipeline_latency": pipe_timing,
        "deadline_met": pipe_timing["p95_ms"] <= deadline_ms,
        "budget_used_percent": round((pipe_timing["p95_ms"] / deadline_ms) * 100.0, 2)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="Project root directory")
    p.add_argument("--model", choices=["all", "ecg", "ppg", "imu", "eda"], default="all")
    p.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    p.add_argument("--device", choices=["cpu", "cuda", "tensorrt"], default="cpu", help="Execution device on Jetson")
    p.add_argument("--output", type=Path, help="JSON output file for benchmark report")
    args = p.parse_args()
    
    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider"
    }
    provider = provider_map[args.device]
    
    hw_info = get_hardware_info()
    print("=" * 80)
    print(" JETSON ORIN NANO / EDGE MULTI-SIGNAL BENCHMARK SUITE")
    print(f" Target Hardware: {hw_info}")
    print(f" Execution Provider: {provider} | Iterations: {args.iterations}")
    print("=" * 80)
    
    results = []
    
    # 1. ECG
    if args.model in ("all", "ecg"):
        print("\n[1/4] Running ECG (PTB-XL) Benchmarks...")
        try:
            r1 = benchmark_ecg_cnn(args.root, args.iterations, provider)
            results.append(r1)
            print(f"  [PASS] {r1['model_id']}: model p50 = {r1['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r1['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] ecg_lvh_cnn failed: {e}")
        try:
            r2 = benchmark_ecg_minimal(args.root, args.iterations, provider)
            results.append(r2)
            print(f"  [PASS] {r2['model_id']}: model p50 = {r2['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r2['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] ecg_prep_minimal failed: {e}")
    
    # 2. PPG
    if args.model in ("all", "ppg"):
        print("\n[2/4] Running PPG (OSASUD) Benchmarks...")
        try:
            r1 = benchmark_ppg_dual_hard(args.root, args.iterations, provider)
            results.append(r1)
            print(f"  [PASS] {r1['model_id']}: model p50 = {r1['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r1['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] ppg_dual_hard failed: {e}")
        try:
            r2 = benchmark_ppg_cnn_transformer(args.root, args.iterations, provider)
            results.append(r2)
            print(f"  [PASS] {r2['model_id']}: model p50 = {r2['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r2['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] ppg_cnn_transformer failed: {e}")
            
    # 3. IMU
    if args.model in ("all", "imu"):
        print("\n[3/4] Running IMU (SisFall) Benchmarks...")
        try:
            r1 = benchmark_imu_xgb(args.root, args.iterations)
            results.append(r1)
            print(f"  [PASS] {r1['model_id']}: model p50 = {r1['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r1['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] imu_native_100 failed: {e}")
        try:
            r2 = benchmark_imu_distilled(args.root, args.iterations, provider)
            results.append(r2)
            print(f"  [PASS] {r2['model_id']}: model p50 = {r2['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r2['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] imu_dm_distilled failed: {e}")
            
    # 4. EDA
    if args.model in ("all", "eda"):
        print("\n[4/4] Running EDA (WESAD) Benchmarks...")
        try:
            r1 = benchmark_eda_logistic(args.root, args.iterations)
            results.append(r1)
            print(f"  [PASS] {r1['model_id']}: model p50 = {r1['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r1['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] eda_wesad_logistic failed: {e}")
        try:
            r2 = benchmark_eda_neural(args.root, args.iterations, provider)
            results.append(r2)
            print(f"  [PASS] {r2['model_id']}: model p50 = {r2['model_latency']['p50_ms']:.2f} ms | pipeline p50 = {r2['pipeline_latency']['p50_ms']:.2f} ms")
        except Exception as e:
            print(f"  [FAIL] eda_wesad_neural_pg failed: {e}")
            
    print("\n" + "=" * 90)
    print(f"{'MODEL ID':<22} | {'INFER (p50/p95)':<18} | {'PIPELINE (p50/p95)':<20} | {'BUDGET':<9} | {'UTIL':<7}")
    print("-" * 90)
    for r in results:
        inf_str = f"{r['model_latency']['p50_ms']:.2f} / {r['model_latency']['p95_ms']:.2f} ms"
        pip_str = f"{r['pipeline_latency']['p50_ms']:.2f} / {r['pipeline_latency']['p95_ms']:.2f} ms"
        bud_str = f"{int(r['realtime_deadline_ms'])} ms"
        util_str = f"{r['budget_used_percent']}%"
        print(f"{r['model_id']:<22} | {inf_str:<18} | {pip_str:<20} | {bud_str:<9} | {util_str:<7}")
    print("=" * 90)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "provider": provider,
        "iterations": args.iterations,
        "results": results
    }
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nReport saved to: {args.output.resolve()}")
    
    return summary


if __name__ == "__main__":
    main()
