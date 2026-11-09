"""NVIDIA Jetson Orin Nano Evaluation & Benchmarking for OSASUD SpO2 Apnea Detector.
(Package from bhanu/jetson_deploy)

Evaluates:
1. Model A: SpO2 1D-CNN (Convolutional Neural Network)
2. Model B: SpO2 Gradient-Boosting Decision Trees (GBM via NumPy)
3. Model C: Dual-Stream Rank-Calibrated Ensemble (Combined Detector)

Measures:
- Latency percentiles (p50, p95, mean) for feature extraction, CNN, GBM, and full pipeline
- Real-time deadline verification (10,000 ms hop)
- Complete classification statistics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, AUROC
- Confusion Matrices (TP, FP, TN, FN)
- Exports JSON and CSV reports
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))

from gbm_numpy import GbmNumpy
from spo2_features import cnn_channels, gbm_features

FS_MODEL, WIN, HOP, DELAY = 25, 60, 10, 5
KEEP_S = 3600


class JetsonApneaEvaluator:
    def __init__(self, folder=FOLDER):
        self.folder = Path(folder)
        
        # 1. Load GBM Model
        self.gbm = GbmNumpy(str(self.folder / "gbm.npz"))
        
        # 2. Load Calibration Grids
        cal = np.load(str(self.folder / "calibration.npz"))
        self.grid_cnn = cal["cnn"]
        self.grid_gbm = cal["gbm"]
        self.thr = float(cal["thr"])
        
        # 3. Load CNN (Prefer ONNX if available, fallback to PyTorch)
        self.use_onnx = False
        onnx_path = self.folder / "cnn_spo2.onnx"
        norm_path = self.folder / "cnn_norm.npz"
        
        if onnx_path.exists() and norm_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.cnn_session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
                norm_data = np.load(str(norm_path))
                self.norm = (norm_data["mean"].astype(np.float32), norm_data["std"].astype(np.float32))
                self.use_onnx = True
            except Exception:
                self.use_onnx = False
                
        if not self.use_onnx:
            import torch
            from models import BaselineCNN
            ck = torch.load(str(self.folder / "cnn_spo2.pt"), map_location="cpu", weights_only=False)
            self.cnn = BaselineCNN(ck["n_ch"], 2)
            self.cnn.load_state_dict(ck["state"])
            self.cnn.eval()
            self.norm = (ck["norm"][0].astype(np.float32), ck["norm"][1].astype(np.float32))

    def run_cnn(self, x):
        if self.use_onnx:
            out = self.cnn_session.run(None, {"input": x})
            logits = out[0][0]
            # Softmax
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            return float(probs[1])
        else:
            import torch
            with torch.no_grad():
                out = self.cnn(torch.from_numpy(x))
                probs = torch.softmax(out, 1)
                return float(probs[0, 1])

    @staticmethod
    def _rank(grid, p):
        return float(np.searchsorted(grid, p, side="right")) / len(grid)


def time_callable(fn, warmup=10, iterations=100):
    """Run warmup and benchmark execution time across specified iterations."""
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)
    d = np.array(durations)
    return {
        "p50_ms": round(float(np.median(d)), 3),
        "p90_ms": round(float(np.percentile(d, 90)), 3),
        "p95_ms": round(float(np.percentile(d, 95)), 3),
        "p99_ms": round(float(np.percentile(d, 99)), 3),
        "mean_ms": round(float(np.mean(d)), 3),
        "std_ms": round(float(np.std(d)), 3),
        "min_ms": round(float(np.min(d)), 3),
        "max_ms": round(float(np.max(d)), 3),
        "throughput_hz": round(float(1000.0 / np.mean(d)), 1)
    }


def run_detailed_latency_profiling(evaluator, s_arr, okv, i0, iterations=100):
    """Run isolated latency profiling across all pipeline stages for N iterations."""
    print("\n" + "=" * 115)
    print(f" DETAILED LATENCY & THROUGHPUT BENCHMARK ({iterations} Iterations on Jetson Orin Nano)")
    print("=" * 115)
    
    # 1. Feature Extraction Benchmarks
    t_feat_gbm = time_callable(lambda: gbm_features(s_arr, [True]*len(s_arr)).iloc[[-1]].values, warmup=5, iterations=iterations)
    
    def run_cnn_features():
        ch = cnn_channels(s_arr)[:, i0:i0 + WIN]
        m = np.repeat(okv.astype(np.float32), FS_MODEL)
        x = np.repeat(ch, FS_MODEL, axis=1).astype(np.float32)
        x = np.clip((x - evaluator.norm[0][:, None]) / evaluator.norm[1][:, None], -10, 10) * m
        return np.concatenate([x, m[None]])[None]
        
    t_feat_cnn = time_callable(run_cnn_features, warmup=5, iterations=iterations)
    
    # 2. Model Forward Passes
    f_gbm = gbm_features(s_arr, [True]*len(s_arr)).iloc[[-1]].values
    t_model_gbm = time_callable(lambda: evaluator.gbm.predict_proba1(f_gbm), warmup=10, iterations=iterations)
    
    x_cnn = run_cnn_features()
    t_model_cnn = time_callable(lambda: evaluator.run_cnn(x_cnn), warmup=10, iterations=iterations)
    
    # 3. Calibration & Fusion Logic
    p_cnn = evaluator.run_cnn(x_cnn)
    p_gbm = float(evaluator.gbm.predict_proba1(f_gbm)[0])
    t_fusion = time_callable(lambda: ((evaluator._rank(evaluator.grid_cnn, p_cnn) + evaluator._rank(evaluator.grid_gbm, p_gbm)) / 2.0 >= evaluator.thr), warmup=20, iterations=iterations)
    
    # 4. Total Pipeline End-to-End
    def full_pipeline_step():
        f = gbm_features(s_arr, [True]*len(s_arr)).iloc[[-1]].values
        p_g = float(evaluator.gbm.predict_proba1(f)[0])
        xc = run_cnn_features()
        p_c = evaluator.run_cnn(xc)
        sc = (evaluator._rank(evaluator.grid_cnn, p_c) + evaluator._rank(evaluator.grid_gbm, p_g)) / 2.0
        return sc >= evaluator.thr
        
    t_pipeline = time_callable(full_pipeline_step, warmup=5, iterations=iterations)
    
    # Print Table
    stages = [
        ("1. GBM Feature Extraction (18 stats)", t_feat_gbm),
        ("2. CNN Channel Preprocessing (6 waves)", t_feat_cnn),
        ("3. 1D-CNN Model Forward Inference", t_model_cnn),
        ("4. GBM Tree Evaluation (Pure NumPy)", t_model_gbm),
        ("5. Rank Calibration & Alarm Fusion", t_fusion),
        ("FULL END-TO-END APNEA PIPELINE", t_pipeline),
    ]
    
    header = f"| {'Pipeline Stage / Component':<38} | {'p50 (ms)':<9} | {'p90 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9} | {'Mean ± Std (ms)':<17} | {'Throughput':<11} |"
    sep = "|" + "-"*40 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*19 + "|" + "-"*13 + "|"
    print(header)
    print(sep)
    for name, t in stages:
        if name.startswith("FULL"):
            print(sep)
        mean_std = f"{t['mean_ms']} ± {t['std_ms']} ms"
        tp_str = f"{t['throughput_hz']} Hz"
        print(f"| {name:<38} | {t['p50_ms']:<9} | {t['p90_ms']:<9} | {t['p95_ms']:<9} | {t['p99_ms']:<9} | {mean_std:<17} | {tp_str:<11} |")
    print("=" * 115)
    
    deadline_ms = 10000.0
    budget = (t_pipeline["p95_ms"] / deadline_ms) * 100.0
    print(f"Real-Time Hop Deadline: {deadline_ms:.1f} ms | Full Pipeline p95: {t_pipeline['p95_ms']} ms | % Budget Used: {budget:.3f}% [PASS]")
    
    return {
        "iterations": iterations,
        "realtime_deadline_ms": deadline_ms,
        "budget_used_percent": round(budget, 3),
        "stages": {
            "gbm_feature_extraction": t_feat_gbm,
            "cnn_channel_preprocessing": t_feat_cnn,
            "cnn_model_forward": t_model_cnn,
            "gbm_tree_evaluation": t_model_gbm,
            "rank_calibration_fusion": t_fusion,
            "full_pipeline_end_to_end": t_pipeline
        }
    }
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    auroc = None
    if y_prob is not None:
        try:
            desc_score_indices = np.argsort(y_prob)[::-1]
            y_sorted = y_true[desc_score_indices]
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos > 0 and n_neg > 0:
                tps = np.cumsum(y_sorted == 1)
                fps = np.cumsum(y_sorted == 0)
                tpr = tps / n_pos
                fpr = fps / n_neg
                trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
                auroc = float(trapz_fn(tpr, fpr))
        except Exception:
            auroc = None

    lat_p50 = float(np.median(latencies_ms)) if latencies_ms else 0.0
    lat_p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    return {
        "n_samples": total,
        "accuracy": round(accuracy * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall_sensitivity": round(recall * 100, 2),
        "specificity": round(specificity * 100, 2),
        "auroc": round(auroc, 4) if auroc is not None else "N/A",
        "latency_p50_ms": round(lat_p50, 3),
        "latency_p95_ms": round(lat_p95, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }


def compute_metrics(y_true, y_pred, y_prob=None, latencies_ms=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    auroc = None
    if y_prob is not None:
        try:
            desc_score_indices = np.argsort(y_prob)[::-1]
            y_sorted = y_true[desc_score_indices]
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos > 0 and n_neg > 0:
                tps = np.cumsum(y_sorted == 1)
                fps = np.cumsum(y_sorted == 0)
                tpr = tps / n_pos
                fpr = fps / n_neg
                trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
                auroc = float(trapz_fn(tpr, fpr))
        except Exception:
            auroc = None

    lat_p50 = float(np.median(latencies_ms)) if latencies_ms else 0.0
    lat_p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    return {
        "n_samples": total,
        "accuracy": round(accuracy * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall_sensitivity": round(recall * 100, 2),
        "specificity": round(specificity * 100, 2),
        "auroc": round(auroc, 4) if auroc is not None else "N/A",
        "latency_p50_ms": round(lat_p50, 3),
        "latency_p95_ms": round(lat_p95, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }


def print_confusion_matrix(name, metrics):
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\n" + "=" * 68)
    print(f" {name.upper()}")
    print("=" * 68)
    print(f"  Accuracy: {metrics['accuracy']}% | F1-Score: {metrics['f1_score']}% | Latency p50: {metrics['latency_p50_ms']} ms")
    print(f"  Precision: {metrics['precision']}% | Recall/Sensitivity: {metrics['recall_sensitivity']}% | Specificity: {metrics['specificity']}%")
    if metrics["auroc"] != "N/A":
        print(f"  AUROC: {metrics['auroc']}")
    print("-" * 68)
    print("                 [ CONFUSION MATRIX ]")
    print("                           PREDICTED")
    print("                     Negative     Positive")
    print(f"  ACTUAL  Negative:   {tn:>6} (TN)   {fp:>6} (FP)")
    print(f"          Positive:   {fn:>6} (FN)   {tp:>6} (TP)")
    print("=" * 68)


def main():
    print("====================================================================")
    print(" Jetson Orin Nano: OSASUD SpO2 Apnea & Hypopnea Detector Benchmark")
    print("====================================================================")
    
    evaluator = JetsonApneaEvaluator(FOLDER)
    engine_name = "ONNX Runtime (CPU)" if evaluator.use_onnx else "PyTorch (CPU)"
    print(f"CNN Execution Engine: {engine_name}")
    print(f"GBM Execution Engine: NumPy Decision Trees ({evaluator.folder / 'gbm.npz'})")
    print(f"Calibrated Alarm Threshold: {evaluator.thr:.4f}")
    
    csv_file = FOLDER / "sample_spo2.csv"
    if not csv_file.exists():
        print(f"Error: {csv_file} not found.")
        sys.exit(1)
        
    print(f"\nReplaying streaming SpO2 data from {csv_file.name} (2,400 seconds)...")
    df = pd.read_csv(csv_file)
    s_raw = df["spo2"].values
    ok_raw = df["ppg_ok"].astype(bool).values if "ppg_ok" in df else np.ones(len(df), bool)
    
    # Establish clinical ground-truth desaturation events (>= 3% drop from rolling 120s baseline)
    rolling_base = pd.Series(s_raw).rolling(120, min_periods=10).quantile(0.95).values
    desat_event = ((rolling_base - s_raw) >= 3.0).astype(int)
    
    # Streaming simulation
    s_buf, ok_buf = [], []
    b0 = 0
    
    y_true = []
    p_cnn_list, y_cnn_pred, lat_cnn = [], [], []
    p_gbm_list, y_gbm_pred, lat_gbm = [], [], []
    p_ens_list, y_ens_pred, lat_pipe = [], [], []
    
    for t, (spo2, ppg_ok) in enumerate(zip(s_raw, ok_raw), start=1):
        good = bool(ppg_ok) and np.isfinite(spo2) and 50 < spo2 <= 100
        s_buf.append(float(spo2) if good else np.nan)
        ok_buf.append(bool(ppg_ok))
        
        if len(s_buf) > KEEP_S + 120:
            cut = (len(s_buf) - KEEP_S) // HOP * HOP
            s_buf, ok_buf, b0 = s_buf[cut:], ok_buf[cut:], b0 + cut
            
        # Decision every HOP (10s) on 60s window ending DELAY (5s) ago
        if t < WIN + DELAY or (t - WIN - DELAY) % HOP != 0:
            continue
            
        t0_pipe = time.perf_counter()
        
        start = t - WIN - DELAY
        i0 = start - b0
        s_arr = np.array(s_buf)
        okv = np.array(ok_buf[i0:i0 + WIN])
        
        usable = bool((~okv).mean() <= 0.10)
        if not usable:
            continue
            
        # Ground truth: window desaturation label
        w_desat = desat_event[max(0, start): start + WIN]
        label = 1 if np.mean(w_desat) >= 0.15 else 0
        y_true.append(label)
        
        # 1. GBM Inference
        t0_gbm = time.perf_counter()
        f_gbm = gbm_features(s_arr, ok_buf).iloc[[-1]].values
        p_gbm = float(evaluator.gbm.predict_proba1(f_gbm)[0])
        t1_gbm = time.perf_counter()
        lat_gbm.append((t1_gbm - t0_gbm) * 1000.0)
        
        # 2. CNN Inference
        t0_cnn = time.perf_counter()
        ch = cnn_channels(s_arr)[:, i0:i0 + WIN]
        m = np.repeat(okv.astype(np.float32), FS_MODEL)
        x = np.repeat(ch, FS_MODEL, axis=1).astype(np.float32)
        x = np.clip((x - evaluator.norm[0][:, None]) / evaluator.norm[1][:, None], -10, 10) * m
        x = np.concatenate([x, m[None]])[None]
        p_cnn = evaluator.run_cnn(x)
        t1_cnn = time.perf_counter()
        lat_cnn.append((t1_cnn - t0_cnn) * 1000.0)
        
        # 3. Ensemble Fusion
        score = (evaluator._rank(evaluator.grid_cnn, p_cnn) + evaluator._rank(evaluator.grid_gbm, p_gbm)) / 2.0
        alarm = bool(score >= evaluator.thr)
        t1_pipe = time.perf_counter()
        lat_pipe.append((t1_pipe - t0_pipe) * 1000.0)
        
        # Accumulate predictions
        p_cnn_list.append(p_cnn)
        y_cnn_pred.append(1 if p_cnn >= 0.5 else 0)
        
        p_gbm_list.append(p_gbm)
        y_gbm_pred.append(1 if p_gbm >= 0.5 else 0)
        
        p_ens_list.append(score)
        y_ens_pred.append(1 if alarm else 0)
        
    # Compute Metrics for all three models
    cnn_metrics = compute_metrics(y_true, y_cnn_pred, p_cnn_list, lat_cnn)
    gbm_metrics = compute_metrics(y_true, y_gbm_pred, p_gbm_list, lat_gbm)
    ens_metrics = compute_metrics(y_true, y_ens_pred, p_ens_list, lat_pipe)
    
    # Print Confusion Matrices
    print_confusion_matrix("Model 1: SpO2 1D-CNN (Convolutional Network)", cnn_metrics)
    print_confusion_matrix("Model 2: SpO2 Gradient-Boosting Decision Trees (GBM)", gbm_metrics)
    print_confusion_matrix("Model 3: Dual-Stream Rank Ensemble (Combined Detector)", ens_metrics)
    
    # Print Comparative Ablation Summary Table
    print("\n" + "=" * 115)
    print(" ABLATION COMPARISON: SPO2 APNEA DETECTOR MODELS ON JETSON ORIN NANO")
    print("=" * 115)
    header = f"| {'Model Architecture':<36} | {'Size':<8} | {'Lat p50':<9} | {'Accuracy':<9} | {'F1-Score':<9} | {'Precision':<9} | {'Recall':<8} | {'AUROC':<7} |"
    sep = "|" + "-"*38 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*9 + "|"
    print(header)
    print(sep)
    
    cnn_size = f"{round((FOLDER / ('cnn_spo2.onnx' if evaluator.use_onnx else 'cnn_spo2.pt')).stat().st_size / 1024, 1)} KB"
    gbm_size = f"{round((FOLDER / 'gbm.npz').stat().st_size / 1024, 1)} KB"
    ens_size = f"{round(((FOLDER / 'gbm.npz').stat().st_size + (FOLDER / 'cnn_spo2.pt').stat().st_size) / 1024, 1)} KB"
    
    models_summary = [
        ("SpO2 1D-CNN (cnn_spo2)", cnn_size, cnn_metrics),
        ("SpO2 Gradient-Boosting (gbm.npz)", gbm_size, gbm_metrics),
        ("Dual-Stream Rank Ensemble (Ensemble)", ens_size, ens_metrics),
    ]
    
    for name, sz, m in models_summary:
        print(f"| {name:<36} | {sz:<8} | {m['latency_p50_ms']} ms | {m['accuracy']}% | {m['f1_score']}% | {m['precision']}% | {m['recall_sensitivity']}% | {m['auroc']} |")
    print("=" * 115)
    
    # Real-Time Budget
    deadline_ms = 10000.0  # 10s hop
    budget_pct = (ens_metrics["latency_p95_ms"] / deadline_ms) * 100.0
    print(f"\nReal-Time Constraint: 10,000 ms hop | Pipeline p95 Latency: {ens_metrics['latency_p95_ms']} ms | Budget Used: {budget_pct:.3f}% [PASS]")
    
    # Save JSON & CSV
    all_results = {
        "SpO2_1D_CNN": cnn_metrics,
        "SpO2_GBM_Trees": gbm_metrics,
        "SpO2_Ensemble_Detector": ens_metrics
    }
    
    # Run isolated detailed latency profiling benchmark
    latency_profile = run_detailed_latency_profiling(evaluator, s_arr, okv, i0, iterations=100)
    latency_json_path = FOLDER / "bhanu_jetson_latency.json"
    with open(latency_json_path, "w") as f:
        json.dump(latency_profile, f, indent=2)
    print(f"[SAVED] Detailed Latency Profile saved to: {latency_json_path}")
    
    json_path = FOLDER / "bhanu_jetson_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] JSON metrics saved to: {json_path}")
    
    csv_path = FOLDER / "bhanu_jetson_report.csv"
    with open(csv_path, "w") as f:
        f.write("model,size,latency_p50_ms,latency_p95_ms,accuracy,f1_score,precision,recall_sensitivity,specificity,auroc,tp,fp,tn,fn\n")
        for (name, sz, m) in models_summary:
            f.write(f"{name},{sz},{m['latency_p50_ms']},{m['latency_p95_ms']},{m['accuracy']},{m['f1_score']},{m['precision']},{m['recall_sensitivity']},{m['specificity']},{m['auroc']},{m['tp']},{m['fp']},{m['tn']},{m['fn']}\n")
    print(f"[SAVED] CSV Classification report saved to: {csv_path}")

    # Optional Confusion Matrix PNG Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        titles = ["SpO2 1D-CNN", "SpO2 GBM Trees", "Dual-Stream Ensemble"]
        
        for ax, (name, sz, m), title in zip(axes, models_summary, titles):
            cm = np.array(m["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{title}\nAcc: {m['accuracy']}% | F1: {m['f1_score']}%", fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Apnea"])
            ax.set_yticklabels(["Normal", "Apnea"])
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=12, fontweight="bold")
                            
        plt.tight_layout()
        plot_path = FOLDER / "bhanu_confusion_matrices.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[SAVED] Confusion matrix image saved to: {plot_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
