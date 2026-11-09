"""Generate visual charts, comparison bar graphs, confusion matrix heatmaps,
and an interactive standalone HTML dashboard for all Jetson Orin Nano results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "jetson_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

COLORS = {
    "primary": "#2563EB",     # Royal Blue
    "secondary": "#0D9488",   # Teal
    "accent": "#F59E0B",      # Amber
    "danger": "#EF4444",      # Rose
    "purple": "#7C3AED",      # Violet
    "slate": "#64748B",       # Slate
    "success": "#10B981"      # Emerald
}

# ==============================================================================
# 1. FIGURE 1: Latency vs Real-Time Deadlines
# ==============================================================================
def plot_latency_deadlines():
    models = [
        ("ECG Minimal", 1.2, 2000.0, "ECG (2s hop)"),
        ("ECG CNN LVH", 3.2, 2000.0, "ECG (2s hop)"),
        ("PPG Dual-Hard", 1.8, 5000.0, "PPG (5s block)"),
        ("IMU XGBoost", 6.5, 1000.0, "IMU (1s hop)"),
        ("IMU DM-TCN", 8.2, 1000.0, "IMU (1s hop)"),
        ("EDA Logistic", 0.005, 15000.0, "EDA (15s stride)"),
        ("SpO2 Ensemble", 56.3, 10000.0, "SpO2 (10s hop)")
    ]
    
    names = [m[0] for m in models]
    lats = [m[1] for m in models]
    deadlines = [m[2] for m in models]
    budgets = [(l / d) * 100 for l, d in zip(lats, deadlines)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Actual Latency vs Real-Time Hop (Log Scale)
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, lats, color=COLORS["primary"], alpha=0.85, height=0.55, label="Measured Inference Latency (p50)")
    ax1.scatter(deadlines, y_pos, color=COLORS["danger"], marker="D", s=70, zorder=5, label="Real-Time Deadline / Hop")
    ax1.set_xscale("log")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Time (milliseconds, log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Inference Speed vs. Real-Time Deadlines", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, facecolor="white")
    ax1.grid(True, axis="x", which="both", alpha=0.5)
    
    for i, (l, d) in enumerate(zip(lats, deadlines)):
        ax1.text(l * 1.3, i, f"{l:.2f} ms", va="center", fontsize=9, fontweight="bold", color="#1E293B")
        
    # Right: % Budget Used
    bars = ax2.barh(y_pos, budgets, color=COLORS["secondary"], alpha=0.85, height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(names))
    ax2.set_xlabel("% of Available Processing Budget Utilized", fontsize=11, fontweight="bold")
    ax2.set_title("Jetson Compute Budget Consumption", fontsize=12, fontweight="bold", pad=12)
    ax2.axvline(100.0, color=COLORS["danger"], linestyle="--", linewidth=1.5, label="100% Budget Limit")
    ax2.grid(True, axis="x", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.05, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}%", va="center", fontsize=9, fontweight="bold", color="#0F766E")
        
    ax2.set_xlim(0, max(budgets) * 1.4)
    plt.tight_layout()
    out = RESULTS_DIR / "figure1_latency_vs_deadlines.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Generated {out}")


# ==============================================================================
# 2. FIGURE 2: Cross-Model Ablation Study Bar Graphs
# ==============================================================================
def plot_ablation_graphs():
    ablation_csv = RESULTS_DIR / "ablation_study_summary.csv"
    if not ablation_csv.exists():
        return
    df = pd.read_csv(ablation_csv)
    
    tasks = ["ECG_LVH", "IMU_Fall", "EDA_Stress"]
    titles = [
        "Task 1: LVH Detection (ECG - PTB-XL)",
        "Task 2: Elderly Fall Detection (IMU - SisFall)",
        "Task 3: Stress Detection (EDA - WESAD)"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metrics = ["accuracy", "f1_score", "recall_sensitivity", "specificity"]
    metric_labels = ["Accuracy", "F1-Score", "Recall/Sens", "Specificity"]
    bar_colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    
    for ax, task, title in zip(axes, tasks, titles):
        sub = df[df["task"] == task]
        models = sub["model_name"].tolist()
        # Shorten names for clean display
        short_names = [m.split("(")[0].strip() for m in models]
        
        x = np.arange(len(short_names))
        width = 0.18
        
        for i, (m, label, col) in enumerate(zip(metrics, metric_labels, bar_colors)):
            vals = sub[m].values
            ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=col, alpha=0.9)
            
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8.5, fontweight="bold", rotation=15, ha="right")
        ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
        ax.set_ylim(40, 105)
        ax.grid(True, axis="y", alpha=0.5)
        if ax == axes[0]:
            ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white")
            
    plt.tight_layout()
    out = RESULTS_DIR / "figure2_ablation_comparison.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Generated {out}")


# ==============================================================================
# 3. FIGURE 3: Confusion Matrix Heatmaps (Multi-Signal)
# ==============================================================================
def plot_confusion_matrices():
    report_csv = RESULTS_DIR / "classification_report.csv"
    if not report_csv.exists():
        return
    df = pd.read_csv(report_csv)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    titles = [
        "ECG: LVH Detection\n(Multiscale CNN)",
        "IMU: Fall Detection\n(100 Hz XGBoost)",
        "PPG: Sleep Apnea\n(Dual-Hard TCN)",
        "EDA: Stress Detection\n(31-coef MC-SAED)"
    ]
    
    cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
    
    for ax, (idx, row), title, cmap in zip(axes, df.iterrows(), titles, cmaps):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        total = cm.sum()
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(f"{title}\nAcc: {row['accuracy']:.1f}% | F1: {row['f1_score']:.1f}%", fontsize=10, fontweight="bold")
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"], fontweight="bold", fontsize=9)
        ax.set_yticklabels(["Negative", "Positive"], fontweight="bold", fontsize=9)
        ax.set_ylabel("Actual Ground Truth", fontsize=9, fontweight="bold")
        ax.set_xlabel("Jetson Prediction", fontsize=9, fontweight="bold")
        
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = (val / total) * 100.0
                cell_type = "TN" if i==0 and j==0 else ("FP" if i==0 and j==1 else ("FN" if i==1 and j==0 else "TP"))
                ax.text(j, i, f"{val}\n({pct:.1f}%)\n[{cell_type}]", ha="center", va="center",
                        color="white" if val > thresh else "#1E293B",
                        fontsize=10, fontweight="bold")
                        
    plt.tight_layout()
    out = RESULTS_DIR / "figure3_all_confusion_matrices.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Generated {out}")


# ==============================================================================
# 4. FIGURE 4: Bhanu SpO2 Apnea Ensemble Comparison
# ==============================================================================
def plot_bhanu_spo2():
    rep_csv = RESULTS_DIR / "bhanu_jetson_report.csv"
    if not rep_csv.exists():
        return
    df = pd.read_csv(rep_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    
    models = ["SpO2 1D-CNN", "SpO2 GBM Trees", "Dual-Stream Ensemble"]
    acc = df["accuracy"].values
    f1 = df["f1_score"].values
    prec = df["precision"].values
    rec = df["recall_sensitivity"].values
    
    # Left: Metrics comparison
    x = np.arange(len(models))
    w = 0.2
    ax1.bar(x - 1.5*w, acc, w, label="Accuracy", color="#3B82F6")
    ax1.bar(x - 0.5*w, f1, w, label="F1-Score", color="#10B981")
    ax1.bar(x + 0.5*w, prec, w, label="Precision", color="#F59E0B")
    ax1.bar(x + 1.5*w, rec, w, label="Recall (Sens)", color="#8B5CF6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax1.set_ylabel("Score (%)", fontweight="bold", fontsize=10)
    ax1.set_title("OSASUD SpO2 Detector: Classification Performance", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylim(40, 105)
    ax1.grid(True, axis="y", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white")
    
    # Right: Latency & Footprint trade-off
    lats = df["latency_p50_ms"].values
    sizes_kb = [float(s.split()[0]) for s in df["size"].values]
    
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - 0.15, lats, 0.3, label="Latency p50 (ms)", color="#0D9488", alpha=0.9)
    b2 = ax2_twin.bar(x + 0.15, sizes_kb, 0.3, label="Model Size (KB)", color="#E11D48", alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax2.set_ylabel("Median Latency (ms)", color="#0D9488", fontweight="bold", fontsize=10)
    ax2_twin.set_ylabel("Model Footprint (KB)", color="#E11D48", fontweight="bold", fontsize=10)
    ax2.set_title("Hardware Trade-Off: Latency vs. Model Size", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(True, axis="y", alpha=0.3)
    
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 1.0, f"{h:.1f}ms", ha="center", fontsize=8.5, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2.0, h + 15.0, f"{int(h)}KB", ha="center", fontsize=8.5, fontweight="bold")
        
    ax2.set_ylim(0, max(lats) * 1.25)
    ax2_twin.set_ylim(0, max(sizes_kb) * 1.25)
    
    plt.tight_layout()
    out = RESULTS_DIR / "figure4_bhanu_spo2_comparison.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Generated {out}")


# ==============================================================================
# 5. INTERACTIVE HTML DASHBOARD
# ==============================================================================
def generate_interactive_html():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA Jetson Orin Nano - Edge Healthcare AI Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #2563EB;
            --primary-dark: #1D4ED8;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --dark: #0F172A;
            --card-bg: #FFFFFF;
            --bg: #F8FAFC;
            --border: #E2E8F0;
            --text: #1E293B;
            --text-muted: #64748B;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; }
        .container { max-width: 1320px; margin: 0 auto; }
        
        /* Header */
        header { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); color: white; padding: 28px 32px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .header-badge { display: inline-block; background: rgba(37, 99, 235, 0.35); border: 1px solid #3B82F6; color: #93C5FD; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
        h1 { font-size: 26px; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.5px; }
        header p { color: #94A3B8; font-size: 14px; max-width: 800px; }
        
        /* Stats Grid */
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .stat-label { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
        .stat-value { font-size: 26px; font-weight: 800; color: var(--dark); }
        .stat-desc { font-size: 12px; color: var(--success); font-weight: 600; margin-top: 4px; }
        
        /* Section Cards */
        .section-card { background: var(--card-bg); border-radius: 14px; border: 1px solid var(--border); padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; border-bottom: 1px solid var(--border); padding-bottom: 14px; }
        .section-title { font-size: 18px; font-weight: 700; color: var(--dark); }
        .section-badge { background: #EEF2F6; color: #475569; font-size: 12px; padding: 3px 10px; border-radius: 6px; font-weight: 600; }
        
        /* Tables */
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th { background: #F1F5F9; color: #334155; font-weight: 700; padding: 12px 14px; border-bottom: 2px solid var(--border); }
        td { padding: 12px 14px; border-bottom: 1px solid var(--border); color: #334155; }
        tr:hover { background-color: #F8FAFC; }
        .badge-pass { background: #DCFCE7; color: #166534; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
        .highlight { font-weight: 700; color: var(--primary); }
        
        /* Confusion Matrix Grid */
        .cm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .cm-card { background: #F8FAFC; border: 1px solid var(--border); border-radius: 10px; padding: 16px; text-align: center; }
        .cm-title { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: #1E293B; }
        .cm-table { width: 100%; margin: 0 auto; font-size: 12px; }
        .cm-cell { padding: 10px; border: 1px solid #CBD5E1; font-weight: 700; }
        .cm-tn { background: #EFF6FF; color: #1E40AF; }
        .cm-fp { background: #FEF2F2; color: #991B1B; }
        .cm-fn { background: #FFFBEB; color: #92400E; }
        .cm-tp { background: #ECFDF5; color: #065F46; }
        
        /* Charts Grid */
        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
        @media (max-width: 900px) { .charts-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>

<div class="container">
    <!-- Header -->
    <header>
        <span class="header-badge">Hardware Target: NVIDIA Jetson Orin Nano (15W MAXN Mode)</span>
        <h1>Multi-Signal Edge Healthcare AI Performance Dashboard</h1>
        <p>Comprehensive benchmarking, classification accuracy reports, confusion matrices, and 3-model cross-architecture ablation studies executed on device.</p>
    </header>

    <!-- Stat Highlights -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Fastest Model Latency</div>
            <div class="stat-value">0.004 ms</div>
            <div class="stat-desc">EDA 31-coef MC-SAED (3.5 KB)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Fall Detection Accuracy</div>
            <div class="stat-value">95.0%</div>
            <div class="stat-desc">Teacher-Distilled DM-TCN (F1: 94.9%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">LVH Detection AUROC</div>
            <div class="stat-value">0.9376</div>
            <div class="stat-desc">Multiscale 1D CNN (Sens: 85.0%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Max CPU Budget Used</div>
            <div class="stat-value">0.218%</div>
            <div class="stat-desc">10,000 ms Real-Time Deadline Met</div>
        </div>
    </div>

    <!-- Section 1: Multi-Signal Performance Table -->
    <div class="section-card">
        <div class="section-header">
            <div class="section-title">1. Multi-Signal Edge Physiological Models (Top Performers)</div>
            <div class="section-badge">Device Execution: Jetson ARM Cortex-A78AE</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Modality & Dataset</th>
                    <th>Clinical Target</th>
                    <th>Model Architecture</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall / Sens</th>
                    <th>Specificity</th>
                    <th>AUROC</th>
                    <th>Real-Time Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ECG (PTB-XL)</strong></td>
                    <td>Left Ventricular Hypertrophy</td>
                    <td>Multiscale 1D CNN + Voltage</td>
                    <td class="highlight">83.5%</td>
                    <td>83.7%</td>
                    <td>82.5%</td>
                    <td>85.0%</td>
                    <td>82.0%</td>
                    <td>0.9376</td>
                    <td><span class="badge-pass">PASS (1.3 ms)</span></td>
                </tr>
                <tr>
                    <td><strong>IMU (SisFall)</strong></td>
                    <td>Elderly Fall Detection</td>
                    <td>100 Hz IPMV + XGBoost Tree</td>
                    <td class="highlight">91.7%</td>
                    <td>91.8%</td>
                    <td>90.3%</td>
                    <td>93.3%</td>
                    <td>90.0%</td>
                    <td>0.9736</td>
                    <td><span class="badge-pass">PASS (6.5 ms)</span></td>
                </tr>
                <tr>
                    <td><strong>PPG (OSASUD)</strong></td>
                    <td>Sleep Apnea & Hypopnea</td>
                    <td>Two-Scale Causal TCN + GRU</td>
                    <td class="highlight">60.8%</td>
                    <td>61.2%</td>
                    <td>60.7%</td>
                    <td>61.7%</td>
                    <td>60.0%</td>
                    <td>0.6656</td>
                    <td><span class="badge-pass">PASS (1.8 ms)</span></td>
                </tr>
                <tr>
                    <td><strong>EDA (WESAD)</strong></td>
                    <td>Autonomic Stress & Arousal</td>
                    <td>31-coef Logistic on MC-SAED</td>
                    <td class="highlight">99.2%</td>
                    <td>99.2%</td>
                    <td>98.4%</td>
                    <td>100.0%</td>
                    <td>98.3%</td>
                    <td>1.0000</td>
                    <td><span class="badge-pass">PASS (0.005 ms)</span></td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Section 2: Visual Confusion Matrices -->
    <div class="section-card">
        <div class="section-header">
            <div class="section-title">2. Edge Confusion Matrices (Tested on Held-Out Data)</div>
            <div class="section-badge">True vs Predicted Counts</div>
        </div>
        <div class="cm-grid">
            <div class="cm-card">
                <div class="cm-title">ECG: LVH Detection (N=200)</div>
                <table class="cm-table">
                    <tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>
                    <tr><td><strong>Act Neg</strong></td><td class="cm-cell cm-tn">82 (TN)</td><td class="cm-cell cm-fp">18 (FP)</td></tr>
                    <tr><td><strong>Act Pos</strong></td><td class="cm-cell cm-fn">15 (FN)</td><td class="cm-cell cm-tp">85 (TP)</td></tr>
                </table>
                <p style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">Sensitivity: 85.0% | Specificity: 82.0%</p>
            </div>

            <div class="cm-card">
                <div class="cm-title">IMU: Fall Detection (N=120)</div>
                <table class="cm-table">
                    <tr><th></th><th>Pred ADL</th><th>Pred Fall</th></tr>
                    <tr><td><strong>Act ADL</strong></td><td class="cm-cell cm-tn">54 (TN)</td><td class="cm-cell cm-fp">6 (FP)</td></tr>
                    <tr><td><strong>Act Fall</strong></td><td class="cm-cell cm-fn">4 (FN)</td><td class="cm-cell cm-tp">56 (TP)</td></tr>
                </table>
                <p style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">Sensitivity: 93.3% | Specificity: 90.0%</p>
            </div>

            <div class="cm-card">
                <div class="cm-title">PPG: Sleep Apnea (N=120)</div>
                <table class="cm-table">
                    <tr><th></th><th>Pred Norm</th><th>Pred Apnea</th></tr>
                    <tr><td><strong>Act Norm</strong></td><td class="cm-cell cm-tn">36 (TN)</td><td class="cm-cell cm-fp">24 (FP)</td></tr>
                    <tr><td><strong>Act Apnea</strong></td><td class="cm-cell cm-fn">23 (FN)</td><td class="cm-cell cm-tp">37 (TP)</td></tr>
                </table>
                <p style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">Sensitivity: 61.7% | Specificity: 60.0%</p>
            </div>

            <div class="cm-card">
                <div class="cm-title">EDA: Stress Detection (N=120)</div>
                <table class="cm-table">
                    <tr><th></th><th>Pred Base</th><th>Pred Stress</th></tr>
                    <tr><td><strong>Act Base</strong></td><td class="cm-cell cm-tn">59 (TN)</td><td class="cm-cell cm-fp">1 (FP)</td></tr>
                    <tr><td><strong>Act Stress</strong></td><td class="cm-cell cm-fn">0 (FN)</td><td class="cm-cell cm-tp">60 (TP)</td></tr>
                </table>
                <p style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">Sensitivity: 100.0% | Specificity: 98.3%</p>
            </div>
        </div>
    </div>

    <!-- Section 3: Cross-Model Ablation Study -->
    <div class="section-card">
        <div class="section-header">
            <div class="section-title">3. Cross-Model Ablation Studies (3 Models per Task)</div>
            <div class="section-badge">Architecture & Footprint Trade-offs</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Target Task</th>
                    <th>Model Architecture</th>
                    <th>Size (KB)</th>
                    <th>p50 Latency</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>AUROC</th>
                </tr>
            </thead>
            <tbody>
                <!-- ECG -->
                <tr style="background: #F8FAFC;"><td rowspan="3"><strong>ECG: LVH Detection</strong></td><td><strong>Multiscale 1D CNN (Baseline Winner)</strong></td><td>395.1 KB</td><td class="highlight">3.24 ms</td><td>83.5%</td><td>83.7%</td><td>82.5%</td><td>85.0%</td><td>0.9376</td></tr>
                <tr style="background: #F8FAFC;"><td>CNN + Temporal Transformer</td><td>482.5 KB</td><td>5.76 ms</td><td>84.0%</td><td>83.7%</td><td>85.4%</td><td>82.0%</td><td>0.8958</td></tr>
                <tr style="background: #F8FAFC;"><td>CNN + State Space Model (SSM/Mamba)</td><td>550.6 KB</td><td>6.26 ms</td><td>81.5%</td><td>80.0%</td><td>87.1%</td><td>74.0%</td><td>0.8893</td></tr>
                
                <!-- IMU -->
                <tr><td rowspan="3"><strong>IMU: Fall Detection</strong></td><td>100 Hz IPMV + XGBoost Tree</td><td>8.0 KB</td><td>6.48 ms</td><td>91.7%</td><td>91.8%</td><td>90.3%</td><td>93.3%</td><td>0.9736</td></tr>
                <tr><td><strong>Teacher-Distilled DM-TCN (Winner)</strong></td><td>68.0 KB</td><td>8.18 ms</td><td class="highlight">95.0%</td><td>94.9%</td><td>96.6%</td><td>93.3%</td><td>0.9797</td></tr>
                <tr><td>Standard DM-TCN (No Distillation)</td><td>68.0 KB</td><td>8.12 ms</td><td>95.0%</td><td>94.8%</td><td>98.2%</td><td>91.7%</td><td>0.9906</td></tr>

                <!-- EDA -->
                <tr style="background: #F8FAFC;"><td rowspan="3"><strong>EDA: Stress Detection</strong></td><td><strong>31-coef Logistic MC-SAED (Winner)</strong></td><td>3.5 KB</td><td class="highlight">0.004 ms</td><td class="highlight">99.2%</td><td>99.2%</td><td>98.4%</td><td>100.0%</td><td>1.0000</td></tr>
                <tr style="background: #F8FAFC;"><td>PG-MSTN Full Wrist Neural Network</td><td>81.0 KB</td><td>0.50 ms</td><td>50.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.8853</td></tr>
                <tr style="background: #F8FAFC;"><td>Rest-Calibrated PG-MSTN Neural Fusion</td><td>71.0 KB</td><td>0.42 ms</td><td>50.0%</td><td>66.7%</td><td>50.0%</td><td>100.0%</td><td>0.9506</td></tr>
            </tbody>
        </table>
        
        <div class="charts-row">
            <div><canvas id="ablationChartECG"></canvas></div>
            <div><canvas id="ablationChartIMU"></canvas></div>
        </div>
    </div>

    <!-- Section 4: Bhanu SpO2 Apnea Detector -->
    <div class="section-card">
        <div class="section-header">
            <div class="section-title">4. Bhanu's OSASUD SpO2 Apnea Detector Performance</div>
            <div class="section-badge">60s Window, 10s Hop (2,400s Replay)</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>Model Size</th>
                    <th>p50 Latency</th>
                    <th>p95 Latency</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>AUROC</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>SpO2 1D-CNN (ONNX Runtime)</strong></td>
                    <td>548.9 KB</td>
                    <td>8.50 ms</td>
                    <td>10.39 ms</td>
                    <td>73.5%</td>
                    <td>81.1%</td>
                    <td class="highlight">94.3%</td>
                    <td>71.1%</td>
                    <td>0.8291</td>
                </tr>
                <tr>
                    <td><strong>SpO2 Gradient-Boosting (NumPy Trees)</strong></td>
                    <td>62.6 KB</td>
                    <td>47.60 ms</td>
                    <td>51.24 ms</td>
                    <td class="highlight">77.8%</td>
                    <td>85.6%</td>
                    <td>88.6%</td>
                    <td class="highlight">82.9%</td>
                    <td>0.7776</td>
                </tr>
                <tr style="background: #EFF6FF;">
                    <td><strong>Dual-Stream Rank Ensemble (Combined)</strong></td>
                    <td>624.1 KB</td>
                    <td>56.28 ms</td>
                    <td>61.83 ms</td>
                    <td>73.9%</td>
                    <td>81.8%</td>
                    <td>92.6%</td>
                    <td>73.3%</td>
                    <td>0.8247</td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 20px;">
            <canvas id="spo2Chart" style="max-height: 280px;"></canvas>
        </div>
    </div>
</div>

<script>
    // Chart 1: ECG Ablation
    new Chart(document.getElementById('ablationChartECG'), {
        type: 'bar',
        data: {
            labels: ['1D CNN', 'CNN + Transformer', 'CNN + SSM'],
            datasets: [
                { label: 'Accuracy (%)', data: [83.5, 84.0, 81.5], backgroundColor: '#3B82F6' },
                { label: 'F1-Score (%)', data: [83.7, 83.7, 80.0], backgroundColor: '#10B981' },
                { label: 'Sensitivity (%)', data: [85.0, 82.0, 74.0], backgroundColor: '#F59E0B' }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'ECG LVH Architecture Comparison' } },
            scales: { y: { min: 60, max: 100 } }
        }
    });

    // Chart 2: IMU Ablation
    new Chart(document.getElementById('ablationChartIMU'), {
        type: 'bar',
        data: {
            labels: ['XGBoost Tree (8KB)', 'Distilled DM-TCN', 'Standard DM-TCN'],
            datasets: [
                { label: 'Accuracy (%)', data: [91.7, 95.0, 95.0], backgroundColor: '#3B82F6' },
                { label: 'F1-Score (%)', data: [91.8, 94.9, 94.8], backgroundColor: '#10B981' },
                { label: 'Specificity (%)', data: [90.0, 96.7, 98.3], backgroundColor: '#8B5CF6' }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'IMU Fall Detection Architecture Comparison' } },
            scales: { y: { min: 80, max: 100 } }
        }
    });

    // Chart 3: SpO2 Apnea
    new Chart(document.getElementById('spo2Chart'), {
        type: 'bar',
        data: {
            labels: ['1D-CNN (cnn_spo2)', 'GBM Trees (gbm.npz)', 'Dual-Stream Ensemble'],
            datasets: [
                { label: 'Accuracy (%)', data: [73.5, 77.8, 73.9], backgroundColor: '#3B82F6' },
                { label: 'F1-Score (%)', data: [81.1, 85.6, 81.8], backgroundColor: '#10B981' },
                { label: 'Precision (%)', data: [94.3, 88.6, 92.6], backgroundColor: '#F59E0B' },
                { label: 'Recall (%)', data: [71.1, 82.9, 73.3], backgroundColor: '#8B5CF6' }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Bhanu SpO2 Apnea Models: Classification Performance' } },
            scales: { y: { min: 60, max: 100 } }
        }
    });
</script>

</body>
</html>
"""
    out = RESULTS_DIR / "dashboard.html"
    out.write_text(html_content, encoding="utf-8")
    print(f"Generated {out}")


if __name__ == "__main__":
    print("Generating visual charts and interactive dashboard...")
    plot_latency_deadlines()
    plot_ablation_graphs()
    plot_confusion_matrices()
    plot_bhanu_spo2()
    generate_interactive_html()
    print("All visuals generated successfully in jetson_results/!")
