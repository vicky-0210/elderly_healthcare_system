# NVIDIA Jetson Orin Nano: Complete Experimental & Deployment Results
**Project: Edge Multi-Signal Physiological Monitoring for Elderly Care**

This directory collects all experimental results, latency benchmarks, classification reports, ablation studies, and visual confusion matrix plots obtained on the **NVIDIA Jetson Orin Nano** (ARM Cortex-A78AE CPU + Ampere 1024-core GPU).

---

## Directory Index

| File | Type | Description |
|---|---|---|
| **`jetson_cpu_benchmark.json`** | JSON | Multi-signal latency profiling (p50, p95, min, max, throughput) and real-time deadline margins. |
| **`classification_report.csv`** | CSV | Tabular classification summary (Acc, Prec, Rec, Spec, F1, AUROC, TP, FP, TN, FN) for all 4 modalities. |
| **`classification_metrics.json`** | JSON | Machine-readable classification metrics and confusion matrix arrays. |
| **`confusion_matrices.png`** | Image | 4-panel visual confusion matrix plot for ECG, IMU, PPG, and EDA edge models. |
| **`ablation_study_summary.csv`** | CSV | Side-by-side comparison of **at least 3 models** each for LVH, Fall, and Stress detection. |
| **`ablation_study_results.json`** | JSON | Detailed ablation study results with latency, parameter size, and classification metrics. |
| **`bhanu_jetson_latency.json`** | JSON | Stage-by-stage latency breakdown (p50, p90, p95, p99, throughput) for Bhanu's OSASUD SpO2 package. |
| **`bhanu_jetson_report.csv`** | CSV | Comparative report for SpO2 CNN vs. GBM Trees vs. Dual-Stream Ensemble. |
| **`bhanu_jetson_metrics.json`** | JSON | Full classification metrics and confusion matrices for the SpO2 apnea models. |
| **`bhanu_confusion_matrices.png`** | Image | 3-panel visual confusion matrix graphic for the SpO2 detector models. |

---

## 1. Multi-Signal Edge Classification & Confusion Matrices

Evaluated on held-out test splits on Jetson Orin Nano:

| Modality & Dataset | Target Clinical Task | Model Architecture | Accuracy | F1-Score | Precision | Recall (Sens) | Specificity | AUROC |
|---|---|---|---|---|---|---|---|---|
| **ECG (PTB-XL)** | Left Ventricular Hypertrophy | Multiscale 1D CNN + Voltage Features | **83.50%** | **83.74%** | 82.52% | 85.00% | 82.00% | **0.9376** |
| **IMU (SisFall)** | Elderly Fall Detection | 100 Hz IPMV + XGBoost Tree Ensemble | **91.67%** | **91.80%** | 90.32% | 93.33% | 90.00% | **0.9739** |
| **PPG (OSASUD)** | Sleep Apnea / Hypopnea | Two-Scale Causal TCN + GRU (Dual-Hard) | **60.83%** | **61.16%** | 60.66% | 61.67% | 60.00% | **0.6656** |
| **EDA (WESAD)** | Autonomic Stress & Arousal | 31-coef Logistic on Causal MC-SAED | **99.17%** | **99.17%** | 98.36% | 100.00% | 98.33% | **0.9833** |

### Confusion Matrix Counts:
* **ECG (LVH)**: TN = 82, FP = 18, FN = 15, TP = 85 (Total: 200 samples)
* **IMU (Falls)**: TN = 54, FP = 6, FN = 4, TP = 56 (Total: 120 trials)
* **PPG (Apnea)**: TN = 36, FP = 24, FN = 23, TP = 37 (Total: 120 windows)
* **EDA (Stress)**: TN = 59, FP = 1, FN = 0, TP = 60 (Total: 120 windows)

---

## 2. Cross-Model Ablation Study Results

Comparison of **3 architectures** per task on Jetson:

### A. LVH Detection (ECG - PTB-XL)
| Model Architecture | Edge Size | p50 Latency | Accuracy | F1-Score | Recall (Sens) | Specificity | AUROC |
|---|---|---|---|---|---|---|---|
| **Multiscale 1D CNN (Baseline Winner)** | **395.1 KB** | **1.253 ms** | **83.50%** | **83.74%** | **85.00%** | 82.00% | **0.9376** |
| **CNN + Temporal Transformer** | 482.5 KB | 2.395 ms | 84.00% | 83.67% | 82.00% | 86.00% | 0.8958 |
| **CNN + State Space Model (SSM/Mamba)** | 550.6 KB | 4.411 ms | 81.50% | 80.00% | 74.00% | **89.00%** | 0.8893 |

### B. Elderly Fall Detection (IMU - SisFall)
| Model Architecture | Edge Size | p50 Latency | Accuracy | F1-Score | Recall (Sens) | Specificity | AUROC |
|---|---|---|---|---|---|---|---|
| **100 Hz IPMV + XGBoost Tree (`native_100`)** | **8.0 KB** | **10.604 ms** | 91.67% | 91.80% | **93.33%** | 90.00% | 0.9739 |
| **Teacher-Distilled DM-TCN (`dm_distilled`)** | 68.0 KB | 14.092 ms | **95.00%** | **94.92%** | **93.33%** | 96.67% | 0.9797 |
| **Standard DM-TCN (No Distillation)** | 68.0 KB | 13.925 ms | **95.00%** | 94.83% | 91.67% | **98.33%** | **0.9906** |

### C. Autonomic Stress Detection (EDA - WESAD)
| Model Architecture | Edge Size | p50 Latency | Accuracy | F1-Score | Recall (Sens) | Specificity | AUROC |
|---|---|---|---|---|---|---|---|
| **31-coef Logistic MC-SAED (`wesad_loso`)** | **3.5 KB** | **0.005 ms** | **99.17%** | **99.17%** | **100.00%** | 98.33% | **0.9833** |
| **PG-MSTN Full Wrist Neural Network** | 81.0 KB | 0.677 ms | 50.00% | 0.00% | 0.00% | **100.00%** | 0.8853 |
| **Rest-Calibrated PG-MSTN Neural Fusion** | 71.0 KB | 0.539 ms | 50.00% | 66.67% | **100.00%** | 0.00% | N/A |

---

## 3. Bhanu's OSASUD SpO2 Apnea Detector (Package from `bhanu/jetson_deploy`)

### Stage-by-Stage Latency Benchmark (100 Iterations on Jetson Orin Nano):
| Pipeline Stage | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean ± Std (ms) | Throughput (Hz) |
|---|---|---|---|---|---|---|
| **1. GBM Feature Extraction (18 stats)** | 8.484 ms | 8.666 ms | 8.695 ms | 9.016 ms | 8.515 ± 0.133 ms | 117.4 Hz |
| **2. CNN Preprocessing (6 channels)** | 2.796 ms | 2.846 ms | 2.978 ms | 3.549 ms | 2.857 ± 0.455 ms | 350.0 Hz |
| **3. 1D-CNN Model Forward (ONNX)** | **0.540 ms** | 0.583 ms | 0.592 ms | 0.676 ms | 0.554 ± 0.031 ms | **1,804.2 Hz** |
| **4. GBM Tree Evaluation (NumPy)** | 7.643 ms | 7.798 ms | 7.897 ms | 8.126 ms | 7.680 ± 0.128 ms | 130.2 Hz |
| **5. Calibration & Alarm Decision** | 0.003 ms | 0.003 ms | 0.003 ms | 0.003 ms | 0.003 ± 0.000 ms | 344,708 Hz |
| **FULL END-TO-END APNEA PIPELINE** | **20.036 ms** | **21.368 ms** | **21.817 ms** | **22.792 ms** | **20.364 ± 0.729 ms** | **49.1 Hz** |

* **Real-Time Constraint**: Hop = **10,000.0 ms** (10 s)
* **p95 Latency**: **21.817 ms**
* **Time Budget Used**: **0.218%** (**PASS**)

### Classification Performance (2,400s Clinical Replay):
| Model | Size | Accuracy | F1-Score | Precision | Recall (Sens) | Specificity | AUROC | Confusion Matrix |
|---|---|---|---|---|---|---|---|---|
| **SpO2 1D-CNN** | 548.9 KB | 73.50% | 81.10% | **94.33%** | 71.12% | **82.98%** | **0.8291** | TN=39, FP=8, FN=54, TP=133 |
| **SpO2 GBM Trees** | **62.6 KB** | **77.78%** | **85.64%** | 88.57% | **82.89%** | 57.45% | 0.7776 | TN=27, FP=20, FN=32, TP=155 |
| **Dual-Stream Ensemble** | 624.1 KB | 73.93% | 81.79% | 92.57% | 73.26% | 76.60% | 0.8248 | TN=36, FP=11, FN=50, TP=137 |
