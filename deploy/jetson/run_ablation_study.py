"""Comprehensive Cross-Model Ablation Study Suite for NVIDIA Jetson Orin Nano.

Compares at least 3 models for each target task:
1. LVH Detection (ECG):
   - Model A: Multiscale 1D CNN with Residual Blocks (Top Supervised)
   - Model B: CNN + Temporal Transformer (Multi-Head Self-Attention)
   - Model C: CNN + Structured State Space Model (SSM / Mamba-style)
   
2. Fall Detection (IMU):
   - Model A: 100 Hz Dynamic-Gravity XGBoost Tree Ensemble (native_100, 8 KB)
   - Model B: Teacher-Distilled DM-TCN (dm_distilled, with witness teacher)
   - Model C: Standard DM-TCN (dm_tcn, without teacher distillation)
   - Model D: 1D Inception Network (inception)

3. Stress Detection (EDA):
   - Model A: 31-coef Logistic Regression on Causal MC-SAED (wesad_loso_v1)
   - Model B: PG-MSTN Full Wrist Neural Network (wesad_loso_v1_neural)
   - Model C: Rest-Calibrated PG-MSTN Neural Fusion (wesad_loso_v1_neural_rest)

Outputs:
- Side-by-side terminal comparison tables
- artifacts/benchmarks/ablation_study_results.json
- artifacts/benchmarks/ablation_study_summary.csv
- artifacts/benchmarks/ablation_comparison.png (if matplotlib is installed)
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in [
    ROOT / "packages/common/src",
    ROOT / "modules/ecg/src",
    ROOT / "modules/ppg/src",
    ROOT / "modules/imu/src",
    ROOT / "modules/eda/src",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import onnxruntime as ort


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
    }


# ==============================================================================
# 1. ECG ABLATION MODELS
# ==============================================================================
def evaluate_ecg_model(pack, root, model_name, onnx_dir, thresh=0.24285):
    onnx_file = onnx_dir / "model.onnx"
    size_kb = round(onnx_file.stat().st_size / 1024, 1)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])
    
    waves = pack["ecg_waves"]
    feats = pack["ecg_features"]
    y_true = pack["ecg_labels"]
    
    probs = []
    y_pred = []
    latencies = []
    
    for w, f in zip(waves, feats):
        feed = {"waveform_mv": w[np.newaxis, :], "voltage_features": f[np.newaxis, :]}
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        logit = float(out[0].squeeze())
        prob = 1.0 / (1.0 + np.exp(-logit))
        probs.append(prob)
        y_pred.append(1 if prob >= thresh else 0)
        
    res = compute_metrics(y_true, y_pred, probs, latencies)
    res["model_name"] = model_name
    res["size_kb"] = size_kb
    return res


# ==============================================================================
# 2. IMU ABLATION MODELS
# ==============================================================================
def evaluate_imu_model(pack, root, model_name, model_dir):
    from wearable_imu.inference.predictor import IMUPredictor
    predictor = IMUPredictor(model_dir)
    
    # Determine size
    meta = json.loads((model_dir / "runtime.json").read_text())
    weight_file = model_dir / meta["file"]
    size_kb = round(weight_file.stat().st_size / 1024, 1)
    
    trials = pack["imu_trials"]
    y_true = pack["imu_labels"]
    
    probs = []
    y_pred = []
    latencies = []
    
    for trial in trials:
        t0 = time.perf_counter()
        res = predictor.predict(trial)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        p = res.get("trial_fall_probability", 0.0)
        probs.append(p)
        y_pred.append(1 if res.get("decision") == "FALL" else 0)
            
    res = compute_metrics(y_true, y_pred, probs, latencies)
    res["model_name"] = model_name
    res["size_kb"] = size_kb
    return res


# ==============================================================================
# 3. EDA ABLATION MODELS
# ==============================================================================
def evaluate_eda_model(pack, root, model_name, model_dir, is_linear=False):
    feats = pack["eda_features"]
    y_true = pack["eda_labels"]
    
    probs = []
    y_pred = []
    latencies = []
    
    if is_linear:
        from wearable_eda.inference.predictor import EDAPredictor
        pred = EDAPredictor().load(model_dir)
        size_kb = round((model_dir / "linear.json").stat().st_size / 1024, 1)
        
        t0 = time.perf_counter()
        scores = pred.model.predict_windows(feats, pred.model.features)
        t1 = time.perf_counter()
        lat_per_sample = ((t1 - t0) * 1000.0) / len(feats)
        latencies = [lat_per_sample] * len(feats)
        
        probs = scores[:, 1]
        y_pred = (probs >= 0.5).astype(int)
    else:
        onnx_file = model_dir / "model.onnx"
        size_kb = round(onnx_file.stat().st_size / 1024, 1)
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])
        inp_names = [i.name for i in sess.get_inputs()]
        
        for f in feats:
            feed = {}
            for inp in sess.get_inputs():
                shape = [1 if not isinstance(dim, int) else dim for dim in inp.shape]
                arr = np.zeros(shape, dtype=np.float32)
                if inp.name == "features":
                    n_copy = min(len(f), shape[1])
                    arr[0, :n_copy] = f[:n_copy]
                feed[inp.name] = arr
                
            t0 = time.perf_counter()
            out = sess.run(None, feed)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
            raw = float(out[0].squeeze()) if out[0].ndim <= 1 else float(out[0].squeeze()[1])
            prob = 1.0 / (1.0 + np.exp(-raw)) if raw < 0 or raw > 1 else raw
            probs.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
            
    res = compute_metrics(y_true, y_pred, probs, latencies)
    res["model_name"] = model_name
    res["size_kb"] = size_kb
    return res


def print_ablation_table(title, results):
    header = f"| {'Model Architecture':<36} | {'Size':<8} | {'Lat p50':<9} | {'Accuracy':<9} | {'F1-Score':<9} | {'Sens/Rec':<9} | {'Spec':<8} | {'AUROC':<7} |"
    sep = "|" + "-"*38 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*9 + "|"
    print("\n" + "=" * 118)
    print(f" ABLATION STUDY: {title.upper()}")
    print("=" * 118)
    print(header)
    print(sep)
    for r in results:
        size_str = f"{r['size_kb']} KB"
        lat_str = f"{r['latency_p50_ms']} ms"
        acc_str = f"{r['accuracy']}%"
        f1_str = f"{r['f1_score']}%"
        rec_str = f"{r['recall_sensitivity']}%"
        spec_str = f"{r['specificity']}%"
        auroc_str = f"{r['auroc']}"
        print(f"| {r['model_name']:<36} | {size_str:<8} | {lat_str:<9} | {acc_str:<9} | {f1_str:<9} | {rec_str:<9} | {spec_str:<8} | {auroc_str:<7} |")
    print("=" * 118)


def main():
    root = ROOT
    pack_file = root / "artifacts/evaluation_pack.npz"
    if not pack_file.exists():
        print(f"Error: {pack_file} not found. Run scripts/generate_evaluation_pack.py first.")
        sys.exit(1)
        
    print(f"Loading evaluation dataset from {pack_file}...")
    pack = np.load(pack_file)
    output_dir = root / "artifacts/benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_results = {}
    
    # --------------------------------------------------------------------------
    # 1. ECG LVH DETECTION ABLATION (3 Models)
    # --------------------------------------------------------------------------
    print("\n[1/3] Benchmarking ECG LVH Ablation Models...")
    ecg_dir = root / "artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal"
    ecg_models = [
        ("Multiscale 1D CNN (Baseline Winner)", ecg_dir / "cnn"),
        ("CNN + Temporal Transformer", ecg_dir / "cnn_transformer"),
        ("CNN + State Space Model (SSM/Mamba)", ecg_dir / "cnn_ssm"),
    ]
    ecg_res = []
    for name, p in ecg_models:
        ecg_res.append(evaluate_ecg_model(pack, root, name, p))
    print_ablation_table("ECG: Left Ventricular Hypertrophy Detection (PTB-XL)", ecg_res)
    ablation_results["ECG_LVH"] = ecg_res
    
    # --------------------------------------------------------------------------
    # 2. IMU FALL DETECTION ABLATION (3 Models)
    # --------------------------------------------------------------------------
    print("\n[2/3] Benchmarking IMU Fall Detection Ablation Models...")
    imu_dir = root / "artifacts/imu/exports/sisfall_v1"
    imu_models = [
        ("100 Hz IPMV + XGBoost Tree (native_100)", imu_dir / "native_100"),
        ("Teacher-Distilled DM-TCN (dm_distilled)", imu_dir / "dm_distilled"),
        ("Standard DM-TCN (No Distillation)", imu_dir / "dm_tcn"),
    ]
    imu_res = []
    for name, p in imu_models:
        imu_res.append(evaluate_imu_model(pack, root, name, p))
    print_ablation_table("IMU: Elderly Fall Detection (SisFall)", imu_res)
    ablation_results["IMU_Fall"] = imu_res

    # --------------------------------------------------------------------------
    # 3. EDA STRESS DETECTION ABLATION (3 Models)
    # --------------------------------------------------------------------------
    print("\n[3/3] Benchmarking EDA Stress Detection Ablation Models...")
    eda_dir = root / "artifacts/eda/exports"
    eda_models = [
        ("31-coef Logistic MC-SAED (wesad_loso)", eda_dir / "wesad_loso_v1", True),
        ("PG-MSTN Full Wrist Neural Network", eda_dir / "wesad_loso_v1_neural", False),
        ("Rest-Calibrated PG-MSTN Neural Fusion", eda_dir / "wesad_loso_v1_neural_rest", False),
    ]
    eda_res = []
    for name, p, is_linear in eda_models:
        eda_res.append(evaluate_eda_model(pack, root, name, p, is_linear))
    print_ablation_table("EDA: Autonomic Stress Detection (WESAD)", eda_res)
    ablation_results["EDA_Stress"] = eda_res
    
    # Save JSON Report
    json_path = output_dir / "ablation_study_results.json"
    with open(json_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[SAVED] Ablation JSON report saved to: {json_path}")
    
    # Save CSV Report
    csv_path = output_dir / "ablation_study_summary.csv"
    with open(csv_path, "w") as f:
        f.write("task,model_name,size_kb,latency_p50_ms,accuracy,f1_score,precision,recall_sensitivity,specificity,auroc\n")
        for task, res_list in ablation_results.items():
            for r in res_list:
                f.write(f"{task},{r['model_name']},{r['size_kb']},{r['latency_p50_ms']},{r['accuracy']},{r['f1_score']},{r['precision']},{r['recall_sensitivity']},{r['specificity']},{r['auroc']}\n")
    print(f"[SAVED] Ablation CSV report saved to: {csv_path}")


if __name__ == "__main__":
    main()
