"""Cross-Modality Classification & Confusion Matrix Evaluation on Jetson Orin Nano.

Evaluates top physiological edge models on held-out test data from artifacts/evaluation_pack.npz:
1. ECG: Left Ventricular Hypertrophy (LVH vs Normal)
2. IMU: Fall Detection (Fall vs ADL)
3. PPG: Sleep Apnea (Apnea Event vs Normal)
4. EDA: Stress Detection (Stress vs Baseline)

Outputs:
- Printed summary tables with Confusion Matrices
- artifacts/benchmarks/classification_metrics.json
- artifacts/benchmarks/classification_report.csv
- artifacts/benchmarks/confusion_matrices.png (if matplotlib is installed)
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project modules are importable
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


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute complete classification metrics: Acc, Prec, Rec, Spec, F1, AUROC, and CM."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Fast trapezoidal AUROC if probabilities supplied
    auroc = None
    if y_prob is not None:
        try:
            # Sort by predicted score
            desc_score_indices = np.argsort(y_prob)[::-1]
            y_sorted = y_true[desc_score_indices]
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos > 0 and n_neg > 0:
                tps = np.cumsum(y_sorted == 1)
                fps = np.cumsum(y_sorted == 0)
                tpr = tps / n_pos
                fpr = fps / n_neg
                # Use numpy trapezoid or trapz
                trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
                auroc = float(trapz_fn(tpr, fpr))
        except Exception:
            auroc = None

    return {
        "n_samples": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall_sensitivity": round(recall * 100, 2),
        "specificity": round(specificity * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "auroc": round(auroc, 4) if auroc is not None else "N/A",
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def print_confusion_matrix(name, task, metrics):
    """Display an aligned ASCII confusion matrix and metrics block."""
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\n" + "=" * 65)
    print(f" {name.upper()}: {task}")
    print("=" * 65)
    print(f"  Test Samples: {metrics['n_samples']} | Accuracy: {metrics['accuracy']}% | F1-Score: {metrics['f1_score']}%")
    print(f"  Precision: {metrics['precision']}% | Recall/Sensitivity: {metrics['recall_sensitivity']}% | Specificity: {metrics['specificity']}%")
    if metrics["auroc"] != "N/A":
        print(f"  AUROC: {metrics['auroc']}")
    print("-" * 65)
    print("                 [ CONFUSION MATRIX ]")
    print("                           PREDICTED")
    print("                     Negative     Positive")
    print(f"  ACTUAL  Negative:   {tn:>6} (TN)   {fp:>6} (FP)")
    print(f"          Positive:   {fn:>6} (FN)   {tp:>6} (TP)")
    print("=" * 65)


def evaluate_ecg(pack, root, provider="CPUExecutionProvider"):
    export_dir = root / "artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal/cnn"
    onnx_file = export_dir / "model.onnx"
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    
    waves = pack["ecg_waves"]
    feats = pack["ecg_features"]
    y_true = pack["ecg_labels"]
    
    thresh = 0.24285
    probs = []
    y_pred = []
    
    for w, f in zip(waves, feats):
        feed = {
            "waveform_mv": w[np.newaxis, :],
            "voltage_features": f[np.newaxis, :]
        }
        out = sess.run(None, feed)
        logit = float(out[0].squeeze())
        prob = 1.0 / (1.0 + np.exp(-logit))
        probs.append(prob)
        y_pred.append(1 if prob >= thresh else 0)
        
    return compute_metrics(y_true, y_pred, probs)


def evaluate_imu(pack, root):
    from wearable_imu.inference.predictor import IMUPredictor
    export_dir = root / "artifacts/imu/exports/sisfall_v1/native_100"
    
    predictor = IMUPredictor(export_dir)
    trials = pack["imu_trials"]
    y_true = pack["imu_labels"]
    
    y_pred = []
    y_prob = []
    for trial in trials:
        res = predictor.predict(trial)
        y_prob.append(res.get("trial_fall_probability", 0.0))
        y_pred.append(1 if res.get("decision") == "FALL" else 0)
        
    return compute_metrics(y_true, y_pred, y_prob)


def evaluate_ppg(pack, root, provider="CPUExecutionProvider"):
    export_dir = root / "artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard"
    onnx_file = export_dir / "model.onnx"
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=[provider])
    
    views = pack["ppg_views"]
    y_true = pack["ppg_labels"]
    
    probs = []
    y_pred = []
    # Calibrated decision threshold on sigmoid probability
    thresh = 0.30
    
    for v in views:
        feed = {"views": v[np.newaxis, :]}
        out = sess.run(None, feed)
        raw_out = out[0].squeeze()
        # Max logit across target seconds converted to probability
        logit = float(raw_out.max()) if raw_out.ndim > 0 else float(raw_out)
        prob = 1.0 / (1.0 + np.exp(-logit))
        probs.append(prob)
        y_pred.append(1 if prob >= thresh else 0)
        
    return compute_metrics(y_true, y_pred, probs)


def evaluate_eda(pack, root):
    from wearable_eda.inference.predictor import EDAPredictor
    export_dir = root / "artifacts/eda/exports/wesad_loso_v1"
    
    pred = EDAPredictor().load(export_dir)
    feats = pack["eda_features"]
    y_true = pack["eda_labels"]
    
    # Linear model prediction returns (N, 2) softmax probabilities
    scores = pred.model.predict_windows(feats, pred.model.features)
    stress_prob = scores[:, 1]
    y_pred = (stress_prob >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, stress_prob)


def main():
    root = ROOT
    pack_file = root / "artifacts/evaluation_pack.npz"
    if not pack_file.exists():
        print(f"Error: {pack_file} not found. Run scripts/generate_evaluation_pack.py first.")
        sys.exit(1)
        
    print(f"Loading test evaluation dataset from: {pack_file}...")
    pack = np.load(pack_file)
    
    output_dir = root / "artifacts/benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. ECG
    print("\nRunning ECG inference on Jetson...")
    ecg_metrics = evaluate_ecg(pack, root)
    print_confusion_matrix("ECG (PTB-XL)", "Left Ventricular Hypertrophy Detection", ecg_metrics)
    all_results["ECG_PTBXL_LVH"] = ecg_metrics
    
    # 2. IMU
    print("\nRunning IMU inference on Jetson...")
    imu_metrics = evaluate_imu(pack, root)
    print_confusion_matrix("IMU (SisFall)", "Elderly Fall Detection", imu_metrics)
    all_results["IMU_SisFall_Fall"] = imu_metrics
    
    # 3. PPG
    print("\nRunning PPG inference on Jetson...")
    ppg_metrics = evaluate_ppg(pack, root)
    print_confusion_matrix("PPG (OSASUD)", "Sleep Apnea & Hypopnea Event Detection", ppg_metrics)
    all_results["PPG_OSASUD_Apnea"] = ppg_metrics
    
    # 4. EDA
    print("\nRunning EDA inference on Jetson...")
    eda_metrics = evaluate_eda(pack, root)
    print_confusion_matrix("EDA (WESAD)", "Autonomic Stress & Arousal Detection", eda_metrics)
    all_results["EDA_WESAD_Stress"] = eda_metrics
    
    # Save structured JSON
    json_path = output_dir / "classification_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] Structured JSON metrics saved to: {json_path}")
    
    # Save CSV Report
    csv_path = output_dir / "classification_report.csv"
    with open(csv_path, "w") as f:
        f.write("modality,task,samples,accuracy,f1_score,precision,recall_sensitivity,specificity,auroc,tp,fp,tn,fn\n")
        for k, v in all_results.items():
            f.write(f"{k},{k},{v['n_samples']},{v['accuracy']},{v['f1_score']},{v['precision']},{v['recall_sensitivity']},{v['specificity']},{v['auroc']},{v['tp']},{v['fp']},{v['tn']},{v['fn']}\n")
    print(f"[SAVED] CSV Classification report saved to: {csv_path}")

    # Optional: Generate PNG confusion matrices if matplotlib is present
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        titles = [
            "ECG: LVH Detection",
            "IMU: Fall Detection",
            "PPG: Sleep Apnea",
            "EDA: Stress Detection"
        ]
        
        for ax, (k, v), title in zip(axes, all_results.items(), titles):
            cm = np.array(v["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{title}\nAcc: {v['accuracy']}% | F1: {v['f1_score']}%", fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(["Negative", "Positive"])
            ax.set_yticklabels(["Negative", "Positive"])
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            
            # Label counts
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=13, fontweight="bold")
                            
        plt.tight_layout()
        plot_path = output_dir / "confusion_matrices.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[SAVED] Confusion matrix plot image saved to: {plot_path}")
    except Exception as e:
        pass


if __name__ == "__main__":
    main()
