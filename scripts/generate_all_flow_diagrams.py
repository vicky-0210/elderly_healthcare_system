"""
generate_all_flow_diagrams.py
Generates four publication-quality black-and-white flow diagrams for:
1. PTB-XL (ECG) - Cardiac Strain / LVH Detection
2. OSASUD (PPG) - Sleep Apnea & Hypopnea Detection
3. SisFall (IMU) - Emergency Fall Detection vs ADL
4. WESAD (EDA) - Stress & Sympathetic Arousal Detection

Strictly Black & White / Grayscale, high resolution (300 DPI), clean vector styling.
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIRS = [
    r"c:\Final Project\elderly_healthcare_system\research\thesis\figures",
    r"c:\Final Project",
    r"C:\Users\vivek\.gemini\antigravity\brain\0c9e1d97-c94f-4911-b79f-e6d420ef1f35"
]

for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

def create_base_canvas(title, subtitle):
    """Creates a 22x12.5 inch landscape canvas with high DPI and clean B&W styling."""
    fig, ax = plt.subplots(figsize=(22, 12.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    # Top Header Banner
    header_box = FancyBboxPatch(
        (1.5, 91.5), 97.0, 7.2,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#000000", edgecolor="#000000", linewidth=1.5
    )
    ax.add_patch(header_box)

    ax.text(50, 96.0, title, color="#FFFFFF", fontsize=16.0, fontweight="bold", ha="center", va="center", family="sans-serif")
    ax.text(50, 93.3, subtitle, color="#E0E0E0", fontsize=10.2, fontweight="normal", ha="center", va="center", family="sans-serif")

    # Footer note
    ax.text(50, 1.0, "Elderly Healthcare Edge System | NVIDIA Jetson Orin Nano Deployment | Strictly Patient-Independent Evaluation | Publication-Quality B&W Flowchart",
            color="#444444", fontsize=8.5, ha="center", va="center", style="italic")

    return fig, ax

def draw_stage_box(ax, x, y, w, h, stage_num, title, subtitle="", bg_color="#FFFFFF", border_color="#000000", border_width=1.8, border_style="-"):
    """Draws a main stage container box with a crisp header."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=bg_color, edgecolor=border_color, linewidth=border_width, linestyle=border_style
    )
    ax.add_patch(box)

    # Header pill for the stage
    pill_h = 4.0
    pill = FancyBboxPatch(
        (x, y + h - pill_h), w, pill_h,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        facecolor="#000000", edgecolor="#000000", linewidth=1.0
    )
    ax.add_patch(pill)

    header_text = f"STAGE {stage_num}: {title.upper()}"
    ax.text(x + w / 2.0, y + h - pill_h / 2.0 + (0.4 if subtitle else 0.0), header_text,
            color="#FFFFFF", fontsize=9.8, fontweight="bold", ha="center", va="center")
    if subtitle:
        ax.text(x + w / 2.0, y + h - pill_h / 2.0 - 1.2, subtitle,
                color="#CCCCCC", fontsize=7.5, ha="center", va="center")

def render_content_lines(ax, x, y_start, w, items, max_width_chars=36, line_h=1.35):
    """Renders structured content items (headings, bullets, key-values, text) with dynamic line wrapping."""
    cur_y = y_start
    for item in items:
        itype = item.get("type", "text")
        if itype == "heading":
            cur_y -= 0.3
            ax.text(x + 0.6, cur_y, item["text"], color="#000000", fontsize=8.2, fontweight="bold", ha="left", va="top")
            cur_y -= (line_h + 0.1)
        elif itype == "bullet":
            text = item["text"]
            wrapped = textwrap.wrap(text, width=max_width_chars)
            for idx, line in enumerate(wrapped):
                if idx == 0:
                    ax.text(x + 0.6, cur_y, "•", color="#000000", fontsize=7.8, fontweight="bold", ha="left", va="top")
                    ax.text(x + 1.5, cur_y, line, color="#111111", fontsize=7.5, fontweight="normal", ha="left", va="top")
                else:
                    ax.text(x + 1.5, cur_y, line, color="#111111", fontsize=7.5, fontweight="normal", ha="left", va="top")
                cur_y -= line_h
            cur_y -= 0.15
        elif itype == "kv":
            k = item["k"]
            v = item["v"]
            full_line = f"{k}: {v}"
            wrapped = textwrap.wrap(full_line, width=max_width_chars)
            for idx, line in enumerate(wrapped):
                if idx == 0:
                    ax.text(x + 0.6, cur_y, "•", color="#000000", fontsize=7.8, fontweight="bold", ha="left", va="top")
                    ax.text(x + 1.5, cur_y, f"{k}:", color="#000000", fontsize=7.5, fontweight="bold", ha="left", va="top")
                    k_offset = len(k) * 0.165 + 1.9
                    if len(wrapped) == 1:
                        ax.text(x + k_offset, cur_y, v, color="#111111", fontsize=7.5, fontweight="normal", ha="left", va="top")
                    else:
                        val_part = line[len(k)+2:]
                        ax.text(x + k_offset, cur_y, val_part, color="#111111", fontsize=7.5, fontweight="normal", ha="left", va="top")
                else:
                    ax.text(x + 1.5, cur_y, line, color="#111111", fontsize=7.5, fontweight="normal", ha="left", va="top")
                cur_y -= line_h
            cur_y -= 0.15
        elif itype == "divider":
            cur_y -= 0.3
            ax.plot([x + 0.6, x + w - 0.6], [cur_y, cur_y], color="#CCCCCC", linewidth=0.7, linestyle="--")
            cur_y -= 0.6
        elif itype == "space":
            cur_y -= item.get("amount", 1.0)
    return cur_y

def draw_sub_card(ax, x, y, w, h, title, items, is_top=False, badge=""):
    """Draws an inner structured card with clear typography."""
    card_bg = "#FFFFFF" if not is_top else "#FAFAFA"
    card_edge = "#000000"
    card_lw = 1.7 if is_top else 1.1
    card_ls = "-" if is_top else "--"

    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor=card_bg, edgecolor=card_edge, linewidth=card_lw, linestyle=card_ls
    )
    ax.add_patch(card)

    # Title bar inside sub-card
    title_bar_h = 2.2
    tb = FancyBboxPatch(
        (x, y + h - title_bar_h), w, title_bar_h,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor="#1A1A1A" if is_top else "#EAEAEA",
        edgecolor=card_edge, linewidth=1.0
    )
    ax.add_patch(tb)

    t_color = "#FFFFFF" if is_top else "#000000"
    ax.text(x + 0.7, y + h - title_bar_h / 2.0, title,
            color=t_color, fontsize=8.6, fontweight="bold", ha="left", va="center")

    if badge:
        ax.text(x + w - 0.7, y + h - title_bar_h / 2.0, badge,
                color=t_color, fontsize=7.2, fontweight="bold", ha="right", va="center")

    # Content with compact line height
    render_content_lines(ax, x, y + h - title_bar_h - 0.5, w, items, max_width_chars=int(w * 2.05), line_h=1.20)

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """Draws a crisp black connecting arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.0,head_width=3.5",
        color="#000000", linewidth=1.6, mutation_scale=1.0
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0 + 0.8
        ax.text(mid_x, mid_y, label, color="#000000", fontsize=7.0, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="#000000", linewidth=0.7))

def draw_branching_arrows(ax, x_start, y_center, x_end, y_targets, labels=None):
    """Draws clean branching arrows from one point to multiple targets."""
    for idx, yt in enumerate(y_targets):
        lbl = labels[idx] if labels and idx < len(labels) else ""
        arrow = FancyArrowPatch(
            (x_start, y_center), (x_end, yt),
            arrowstyle="-|>,head_length=5.0,head_width=3.5",
            color="#000000", linewidth=1.5, mutation_scale=1.0
        )
        ax.add_patch(arrow)
        if lbl:
            mid_x = (x_start + x_end) / 2.0
            mid_y = (y_center + yt) / 2.0 + (0.8 if yt >= y_center else -0.8)
            ax.text(mid_x, mid_y, lbl, color="#000000", fontsize=6.8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="#000000", linewidth=0.6))

def draw_converging_arrows(ax, x_starts, y_sources, x_end, y_center):
    """Draws converging arrows from multiple models to the output stage."""
    for idx, ys in enumerate(y_sources):
        arrow = FancyArrowPatch(
            (x_starts, ys), (x_end, y_center),
            arrowstyle="-|>,head_length=5.0,head_width=3.5",
            color="#000000", linewidth=1.5, mutation_scale=1.0
        )
        ax.add_patch(arrow)


# ==============================================================================
# 1. ECG FLOW DIAGRAM (PTB-XL DATASET)
# ==============================================================================
def generate_ecg_flow():
    fig, ax = create_base_canvas(
        title="END-TO-END PHYSIOLOGICAL PIPELINE: CARDIAC STRAIN & LVH PATTERN DETECTION",
        subtitle="Modality: Single-Lead Electrocardiogram (ECG) | Dataset: PTB-XL (v1.0.3) | Hardware: BioAmp EXG Pill -> NVIDIA Jetson Orin Nano"
    )

    # 1. Stage 1: Acquisition (x: 1.5, w: 16.5)
    draw_stage_box(ax, 1.5, 3.5, 16.5, 86.0, 1, "Sensor & Signal", "Physical Wearable Hardware")
    s1_items = [
        {"type": "heading", "text": "Hardware Sensor:"},
        {"type": "kv", "k": "Sensor", "v": "BioAmp EXG Pill (Upside Down Labs)"},
        {"type": "kv", "k": "Electrode Config", "v": "3-Electrode Lead-II approximation"},
        {"type": "bullet", "text": "RA (-): Right Arm / Upper Chest"},
        {"type": "bullet", "text": "LL (+): Left Leg / Lower Left Rib"},
        {"type": "bullet", "text": "RL (Ref): Right Leg ground reference"},
        {"type": "divider"},
        {"type": "heading", "text": "Signal Specifications:"},
        {"type": "kv", "k": "Modality", "v": "Lead-II Electrocardiogram (ECG)"},
        {"type": "kv", "k": "Physical Units", "v": "Calibrated millivolts (mV)"},
        {"type": "kv", "k": "ADC Resolution", "v": "12-bit successive approx ADC"},
        {"type": "kv", "k": "Source Rate", "v": "fs = 500 Hz (PTB-XL dataset)"},
        {"type": "kv", "k": "Target Stream", "v": "fs = 360 Hz (edge wearable)"},
        {"type": "kv", "k": "Buffer Window", "v": "10.0 seconds (3,600 samples)"},
        {"type": "kv", "k": "Hop Cadence", "v": "2.0 s sliding hop (80% overlap)"},
        {"type": "divider"},
        {"type": "heading", "text": "Physical Calibration:"},
        {"type": "bullet", "text": "Linear hardware count-to-mV mapping:"},
        {"type": "bullet", "text": "  V_mV = (count - offset) * (V_ref/4096) / Gain"},
        {"type": "bullet", "text": "CRITICAL: Absolute millivolts strictly preserved; no whole-file z-scoring!"}
    ]
    render_content_lines(ax, 1.5, 84.5, 16.5, s1_items, max_width_chars=34, line_h=1.35)

    # 2. Stage 2: Dataset & Detection Target (x: 19.2, w: 17.0)
    draw_stage_box(ax, 19.2, 3.5, 17.0, 86.0, 2, "Dataset & Target", "Clinical Benchmark & Task")
    s2_items = [
        {"type": "heading", "text": "Public Benchmark Dataset:"},
        {"type": "kv", "k": "Dataset", "v": "PTB-XL v1.0.3 (PhysioNet)"},
        {"type": "kv", "k": "Clinical Scope", "v": "Clinical 12-lead ECG corpus"},
        {"type": "kv", "k": "Total Records", "v": "21,798 ten-second clinical ECGs"},
        {"type": "kv", "k": "Patient Count", "v": "18,885 unique patients"},
        {"type": "kv", "k": "Cohort Balance", "v": "2,131 LVH statements vs 19,667 rest"},
        {"type": "divider"},
        {"type": "heading", "text": "Patient-Independent Split:"},
        {"type": "bullet", "text": "Official 10-fold patient assignment:"},
        {"type": "bullet", "text": "  • Folds 1-6 (60%): Model development"},
        {"type": "bullet", "text": "  • Folds 7-8 (20%): Validation & early stop"},
        {"type": "bullet", "text": "  • Fold 9 (10%): Platt calibration & cutoff"},
        {"type": "bullet", "text": "  • Fold 10 (10%): Strictly frozen test fold"},
        {"type": "bullet", "text": "Zero patient leakage across splits!"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Detection Target:"},
        {"type": "kv", "k": "Condition", "v": "Left Ventricular Hypertrophy (LVH)"},
        {"type": "kv", "k": "Pathology", "v": "Myocardial thickening from chronic hypertension & aging"},
        {"type": "kv", "k": "Clinical Risk", "v": "Precursor to heart failure & stroke"},
        {"type": "kv", "k": "Ground Truth", "v": "Cardiologist verified SCP statements (Sokolow-Lyon & Cornell criteria)"}
    ]
    render_content_lines(ax, 19.2, 84.5, 17.0, s2_items, max_width_chars=34, line_h=1.35)

    # 3. Stage 3: Causal Preprocessing Pipeline (x: 37.4, w: 20.0)
    draw_stage_box(ax, 37.4, 3.5, 20.0, 86.0, 3, "Preprocessing Pipeline", "Deterministic Causal DSP")
    s3_items = [
        {"type": "heading", "text": "NOVELTY: Voltage Preservation"},
        {"type": "bullet", "text": "Standard ML z-scores ECG, destroying absolute R-wave amplitude (primary LVH biomarker). Our causal FIR preserves calibrated millivolts."},
        {"type": "divider"},
        {"type": "heading", "text": "Step 1: Rational Polyphase FIR Resampler"},
        {"type": "kv", "k": "Conversion", "v": "500 Hz -> 360 Hz (exact 18/25 ratio)"},
        {"type": "bullet", "text": "Upsample L=18 -> Kaiser LPF -> Decimate M=25"},
        {"type": "bullet", "text": "Kaiser window (beta=5.0, fc=162 Hz, atten > 65 dB)"},
        {"type": "bullet", "text": "Preserves 10.0 s (3,600 samples) in true mV"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 2: Causal Quality & Integrity Gate"},
        {"type": "bullet", "text": "Finite check: Rejects NaN / Inf corrupted frames"},
        {"type": "bullet", "text": "Clock jitter check: |dt - 1/fs| <= 0.2/fs"},
        {"type": "bullet", "text": "Flatline check: std(V) < 10^-5 mV"},
        {"type": "bullet", "text": "ADC rail clipping check: dwell at upper/lower rails"},
        {"type": "bullet", "text": "Invalid windows bypass neural execution entirely"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 3: Multi-Branch Morphology Views"},
        {"type": "bullet", "text": "Branch A: Calibrated waveform (3,600 x 1)"},
        {"type": "bullet", "text": "Branch B: Train-standardized voltage features (QRS amplitude, ST-segment elevation/depression)"}
    ]
    render_content_lines(ax, 37.4, 84.5, 20.0, s3_items, max_width_chars=39, line_h=1.35)

    # 4. Stage 4: Top Three Models (x: 58.6, w: 24.5)
    draw_stage_box(ax, 58.6, 3.5, 24.5, 86.0, 4, "Top Three Models", "Evaluated on Held-Out Patient Fold 10")

    # Model 1 Card (y: 58.0, h: 25.8)
    m1_items = [
        {"type": "kv", "k": "Architecture", "v": "Multiscale 1D Residual Temporal CNN"},
        {"type": "bullet", "text": "Multiscale conv stem (k=3, 7; dilations 1, 2) + residual skips"},
        {"type": "bullet", "text": "Global temporal mean & max pooling with calibrated sigmoid"},
        {"type": "kv", "k": "Complexity", "v": "94,753 trainable parameters (382 KB)"},
        {"type": "heading", "text": "Held-Out Fold 10 Test Results:"},
        {"type": "bullet", "text": "AUROC: 0.9329 +/- 0.0044  |  AUPRC: 0.8838 +/- 0.0057"},
        {"type": "bullet", "text": "F1-Score: 0.7752 +/- 0.0130  |  Brier Score: 0.0674"},
        {"type": "bullet", "text": "Sensitivity (Recall): 82.24%  |  Specificity: 90.84%"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "3.60 ms forward (27.1 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 58.0, 22.7, 25.8, "1. Multiscale 1D ResCNN (ecg_lvh_cnn)", m1_items, is_top=True, badge="TOP PERFORMER")

    # Model 2 Card (y: 31.0, h: 25.8)
    m2_items = [
        {"type": "kv", "k": "Architecture", "v": "CNN + Multi-Head Temporal Transformer"},
        {"type": "bullet", "text": "Multiscale 1D conv stem + sinusoidal positional encoding"},
        {"type": "bullet", "text": "Multi-head temporal self-attention transformer blocks"},
        {"type": "bullet", "text": "Captures long-range interbeat cardiac rhythm dependencies"},
        {"type": "kv", "k": "Complexity", "v": "92,835 trainable parameters (375 KB)"},
        {"type": "heading", "text": "Held-Out Fold 10 Test Results:"},
        {"type": "bullet", "text": "AUROC: 0.9102 +/- 0.0030  |  AUPRC: 0.8507 +/- 0.0046"},
        {"type": "bullet", "text": "F1-Score: 0.7487 +/- 0.0066  |  Brier Score: 0.0777"},
        {"type": "bullet", "text": "Sensitivity (Recall): 77.10%  |  Specificity: 91.18%"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "3.82 ms forward pass on Jetson CPU"}
    ]
    draw_sub_card(ax, 59.5, 31.0, 22.7, 25.8, "2. CNN-Transformer (cnn_transformer)", m2_items, is_top=False, badge="TEMPORAL ATTENTION")

    # Model 3 Card (y: 4.0, h: 25.8)
    m3_items = [
        {"type": "kv", "k": "Architecture", "v": "Privileged Distilled ResNet (prep_minimal)"},
        {"type": "bullet", "text": "1D Residual student trained via Knowledge Distillation"},
        {"type": "bullet", "text": "12-lead Inception teacher (E2_inception) transfers multi-lead spatial context into lightweight edge student"},
        {"type": "kv", "k": "Complexity", "v": "9,450 parameters (42 KB ONNX graph)"},
        {"type": "heading", "text": "Held-Out Fold 10 Test Results (Broad Cohort):"},
        {"type": "bullet", "text": "AUROC: 0.8040  |  AUPRC: 0.4373  |  F1: 0.4289"},
        {"type": "bullet", "text": "Sensitivity: 51.49%  |  Specificity: 89.48%"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.43 ms forward (0.90 ms E2E; 0.05% budget!)"}
    ]
    draw_sub_card(ax, 59.5, 4.0, 22.7, 25.8, "3. Distilled Student (prep_minimal)", m3_items, is_top=False, badge="EDGE KD WINNER")

    # 5. Stage 5: Output Decision & Alert Generation (x: 84.3, w: 14.2)
    draw_stage_box(ax, 84.3, 3.5, 14.2, 86.0, 5, "Decision & Alert", "Inference & Edge Output")
    s5_items = [
        {"type": "heading", "text": "Probability Calibration:"},
        {"type": "bullet", "text": "Fold-9 fitted Platt scaling: p = sigma(a * z + b)"},
        {"type": "bullet", "text": "Prevents neural overconfidence"},
        {"type": "divider"},
        {"type": "heading", "text": "Decision Logic:"},
        {"type": "bullet", "text": "Operating cutoff: tau = 0.42 (selected for >= 90% specificity)"},
        {"type": "bullet", "text": "Hysteresis filter: 2 consecutive positive 10 s windows required for alert"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Outputs:"},
        {"type": "bullet", "text": "Status: LVH_DETECTED vs NORMAL_SINUS"},
        {"type": "bullet", "text": "Risk Score: p(LVH) in [0, 1]"},
        {"type": "bullet", "text": "Quality Flag: VALID / REJECT"},
        {"type": "divider"},
        {"type": "heading", "text": "Edge Budget (Jetson):"},
        {"type": "kv", "k": "E2E Latency", "v": "0.90 - 27.1 ms"},
        {"type": "kv", "k": "Hop Deadline", "v": "2,000 ms (2 s hop)"},
        {"type": "kv", "k": "Budget Consumed", "v": "0.05% - 1.48%"},
        {"type": "kv", "k": "Deadline Status", "v": "REAL-TIME PASS"}
    ]
    render_content_lines(ax, 84.3, 84.5, 14.2, s5_items, max_width_chars=27, line_h=1.35)

    # Connecting Arrows
    draw_arrow(ax, 18.0, 46.5, 19.2, 46.5, "Calibrated mV")
    draw_arrow(ax, 36.2, 46.5, 37.4, 46.5, "Lead-II Folds")

    # Branching arrows from Stage 3 to Stage 4 Models
    draw_branching_arrows(ax, 57.4, 46.5, 59.5, [70.9, 43.9, 16.9], ["(1) Raw+Volt", "(2) Waveform", "(3) Resampled"])

    # Converging arrows from Stage 4 Models to Stage 5 Decision
    draw_converging_arrows(ax, 82.2, [70.9, 43.9, 16.9], 84.3, 46.5)

    # Save to all target paths
    filename = "flow_ptbxl_ecg_lvh.png"
    for out_dir in OUTPUT_DIRS:
        dest_path = os.path.join(out_dir, filename)
        fig.savefig(dest_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Saved: {dest_path}")
    plt.close(fig)


# ==============================================================================
# 2. PPG FLOW DIAGRAM (OSASUD DATASET)
# ==============================================================================
def generate_ppg_flow():
    fig, ax = create_base_canvas(
        title="END-TO-END PHYSIOLOGICAL PIPELINE: SLEEP APNEA & HYPOPNEA DETECTION",
        subtitle="Modality: Optical Photoplethysmogram (PPG) | Dataset: OSASUD | Hardware: MAX30102 Pulse Oximeter -> NVIDIA Jetson Orin Nano"
    )

    # 1. Stage 1: Acquisition (x: 1.5, w: 16.5)
    draw_stage_box(ax, 1.5, 3.5, 16.5, 86.0, 1, "Sensor & Signal", "Optical Wearable Hardware")
    s1_items = [
        {"type": "heading", "text": "Hardware Sensor:"},
        {"type": "kv", "k": "Sensor", "v": "MAX30102 Pulse Oximeter"},
        {"type": "kv", "k": "Technology", "v": "Dual-wavelength reflectometry"},
        {"type": "bullet", "text": "Red LED: 660 nm (oxygenated Hb)"},
        {"type": "bullet", "text": "Infrared LED: 880 nm (deoxygenated Hb)"},
        {"type": "bullet", "text": "Integrated 18-bit Low-Noise ADC"},
        {"type": "bullet", "text": "Ambient light cancellation circuitry"},
        {"type": "divider"},
        {"type": "heading", "text": "Signal Specifications:"},
        {"type": "kv", "k": "Modality", "v": "Optical PPG & Derived SpO2"},
        {"type": "kv", "k": "Native Rate", "v": "fs = 80 Hz / 100 Hz"},
        {"type": "kv", "k": "Signal Type", "v": "Continuous pulsatile AC + baseline DC"},
        {"type": "kv", "k": "Streaming Buffer", "v": "30 s context with 5 s hop"},
        {"type": "kv", "k": "History Buffer", "v": "Up to 90 s modulation history"},
        {"type": "divider"},
        {"type": "heading", "text": "Hardware Interface:"},
        {"type": "bullet", "text": "Fast I2C bus @ 400 kHz (smbus2 Linux driver)"},
        {"type": "bullet", "text": "32-sample hardware FIFO interrupt queue"},
        {"type": "bullet", "text": "Automatic LED current feedback loop prevents sensor saturation across skin tones"}
    ]
    render_content_lines(ax, 1.5, 84.5, 16.5, s1_items, max_width_chars=34, line_h=1.35)

    # 2. Stage 2: Dataset & Detection Target (x: 19.2, w: 17.0)
    draw_stage_box(ax, 19.2, 3.5, 17.0, 86.0, 2, "Dataset & Target", "Clinical Benchmark & Task")
    s2_items = [
        {"type": "heading", "text": "Public Benchmark Dataset:"},
        {"type": "kv", "k": "Dataset", "v": "OSASUD Public Database"},
        {"type": "kv", "k": "Clinical Setting", "v": "Umea University Stroke Unit"},
        {"type": "kv", "k": "Patient Cohort", "v": "30 overnight hospital PSG recordings"},
        {"type": "kv", "k": "Total Scored Time", "v": "821,545 seconds (228.2 hours)"},
        {"type": "kv", "k": "Reference Events", "v": "5,985 annotated event intervals"},
        {"type": "divider"},
        {"type": "heading", "text": "Patient-Independent Split:"},
        {"type": "bullet", "text": "Rigid 5-fold cross-validation:"},
        {"type": "bullet", "text": "  • 18 patients: Model training"},
        {"type": "bullet", "text": "  • 6 patients: Threshold & calibration"},
        {"type": "bullet", "text": "  • 6 patients: Held-out test evaluation"},
        {"type": "bullet", "text": "Every patient appears in test exactly once"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Detection Target:"},
        {"type": "kv", "k": "Condition", "v": "Obstructive Sleep Apnea & Hypopnea"},
        {"type": "kv", "k": "Pathophysiology", "v": "Cessation (apnea >= 90%) or reduction (hypopnea >= 30%) in airflow >= 10 s"},
        {"type": "kv", "k": "Clinical Risks", "v": "Nocturnal desaturation, hypertension, cardiac arrhythmia, sleep fragmentation"},
        {"type": "kv", "k": "Ground Truth", "v": "Board-certified sleep physician PSG scoring"}
    ]
    render_content_lines(ax, 19.2, 84.5, 17.0, s2_items, max_width_chars=34, line_h=1.35)

    # 3. Stage 3: Causal Preprocessing Pipeline (x: 37.4, w: 20.0)
    draw_stage_box(ax, 37.4, 3.5, 20.0, 86.0, 3, "Preprocessing Pipeline", "Deterministic Causal DSP")
    s3_items = [
        {"type": "heading", "text": "NOVELTY: Mean-Preserving Haar Shrinkage"},
        {"type": "bullet", "text": "Standard filters remove baseline wander, discarding respiratory modulation. Our pipeline splits optical traces into pulse and slow modulation views, using Haar shrinkage that attenuates sensor noise while strictly preserving the envelope (+0.041 AP gain!)."},
        {"type": "divider"},
        {"type": "heading", "text": "Step 1: Haar Wavelet Denoising"},
        {"type": "bullet", "text": "Finest-level detail coefficient shrunk by 0.5 * MAD"},
        {"type": "bullet", "text": "Preserves pairwise sample mean: (x1 + x2)/2 intact"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 2: Multi-View Extraction (31 feats/s)"},
        {"type": "bullet", "text": "View 1: Pulse Morphology (16 feats):"},
        {"type": "bullet", "text": "  • 0.45-5.5 Hz 2nd-order Butterworth filter"},
        {"type": "bullet", "text": "  • 16 shape coords/s normalized by 45 s pulse ref"},
        {"type": "bullet", "text": "View 2: Respiratory Modulation (11 feats):"},
        {"type": "bullet", "text": "  • 0.08-0.38 Hz respiratory envelope filter"},
        {"type": "bullet", "text": "  • Baseline drift, amplitude var, IBI var, RMS"},
        {"type": "bullet", "text": "View 3: Signal Quality & Integrity (4 feats):"},
        {"type": "bullet", "text": "  • HF residual RMS, equal-neighbor fraction, peak staleness, ADC rail dwell fraction"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 3: Robust Compression"},
        {"type": "bullet", "text": "Median/IQR robust scaling + asinh compression"}
    ]
    render_content_lines(ax, 37.4, 84.5, 20.0, s3_items, max_width_chars=39, line_h=1.35)

    # 4. Stage 4: Top Three Models (x: 58.6, w: 24.5)
    draw_stage_box(ax, 58.6, 3.5, 24.5, 86.0, 4, "Top Three Models", "Evaluated on Held-Out OSASUD Patients")

    # Model 1 Card (y: 58.0, h: 25.8)
    m1_items = [
        {"type": "kv", "k": "Architecture", "v": "Dual-Rate Causal TCN + GRU (dual_hard)"},
        {"type": "bullet", "text": "Two time-scales: 12 s pulse waveform branch (causal conv) + 90 s modulation branch (dilated TCN blocks, dilations 2, 4, 8)"},
        {"type": "bullet", "text": "24-unit unidirectional GRU recurrent layer"},
        {"type": "kv", "k": "Complexity", "v": "9,097 trainable parameters (71.6 KB ONNX)"},
        {"type": "heading", "text": "Held-Out Patient Test Results:"},
        {"type": "bullet", "text": "Mean Patient AP: 0.247  (vs 0.160 constant baseline)"},
        {"type": "bullet", "text": "Event F1-Score: 0.257  |  Event Recall: 46.3%"},
        {"type": "bullet", "text": "False Events: 56.2/hour (17.7 fewer false alerts/h!)"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "1.17 ms forward (15.91 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 58.0, 22.7, 25.8, "1. Dual-Rate TCN+GRU (dual_hard)", m1_items, is_top=True, badge="TOP WINNER")

    # Model 2 Card (y: 31.0, h: 25.8)
    m2_items = [
        {"type": "kv", "k": "Architecture", "v": "Multiscale CNN + Transformer"},
        {"type": "bullet", "text": "Multiscale 1D temporal conv layers extracting 30 s context"},
        {"type": "bullet", "text": "Multi-head temporal self-attention transformer blocks"},
        {"type": "bullet", "text": "Simultaneously outputs 5 individual second event logits"},
        {"type": "kv", "k": "Complexity", "v": "25,573 trainable parameters (104 KB)"},
        {"type": "heading", "text": "Held-Out Patient Test Results:"},
        {"type": "bullet", "text": "Mean Patient AP: 0.238  |  Event Recall: 52.2%"},
        {"type": "bullet", "text": "Event F1-Score: 0.252  |  False Events: 68.8/hour"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.50 ms forward (13.96 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 31.0, 22.7, 25.8, "2. CNN-Transformer (cnn_transformer)", m2_items, is_top=False, badge="ATTENTION CONTEXT")

    # Model 3 Card (y: 4.0, h: 25.8)
    m3_items = [
        {"type": "kv", "k": "Architecture", "v": "Single-Scale Binary TCN + GRU"},
        {"type": "bullet", "text": "30 s single-scale causal dilated TCN temporal backbone"},
        {"type": "bullet", "text": "Residual temporal blocks feeding a recurrent GRU layer"},
        {"type": "kv", "k": "Complexity", "v": "18,400 parameters (76 KB checkpoint)"},
        {"type": "heading", "text": "Held-Out Patient Test Results:"},
        {"type": "bullet", "text": "Mean Patient AP: 0.242  |  Event Recall: 57.6% (up to 62.4%)"},
        {"type": "bullet", "text": "Event F1-Score: 0.276  |  False Events: 68.2/hour"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "1.05 ms forward pass on Jetson CPU"}
    ]
    draw_sub_card(ax, 59.5, 4.0, 22.7, 25.8, "3. Single-Scale TCN+GRU (single_scale)", m3_items, is_top=False, badge="HIGH RECALL")

    # 5. Stage 5: Output Decision & Alert Generation (x: 84.3, w: 14.2)
    draw_stage_box(ax, 84.3, 3.5, 14.2, 86.0, 5, "Decision & Alert", "Inference & Edge Output")
    s5_items = [
        {"type": "heading", "text": "Boundary Decoding:"},
        {"type": "bullet", "text": "Emits 5 separate second predictions per 5 s block"},
        {"type": "bullet", "text": "Preserves exact onset boundaries without whole-window blurring"},
        {"type": "divider"},
        {"type": "heading", "text": "Event Confirmation:"},
        {"type": "bullet", "text": "Validation-fitted F1 threshold"},
        {"type": "bullet", "text": "Persistence: >= 5 consecutive positive seconds to emit event"},
        {"type": "bullet", "text": "Temporal IoU >= 0.30 matching"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Outputs:"},
        {"type": "bullet", "text": "Status: APNEA / NORMAL"},
        {"type": "bullet", "text": "Apnea-Hypopnea Index (AHI)"},
        {"type": "bullet", "text": "Nocturnal Desaturation Burdens"},
        {"type": "divider"},
        {"type": "heading", "text": "Edge Budget (Jetson):"},
        {"type": "kv", "k": "E2E Latency", "v": "13.9 - 22.4 ms"},
        {"type": "kv", "k": "Block Deadline", "v": "5,000 ms (5 s block)"},
        {"type": "kv", "k": "Budget Consumed", "v": "0.41% - 0.45%"},
        {"type": "kv", "k": "Deadline Status", "v": "REAL-TIME PASS"}
    ]
    render_content_lines(ax, 84.3, 84.5, 14.2, s5_items, max_width_chars=27, line_h=1.35)

    # Connecting Arrows
    draw_arrow(ax, 18.0, 46.5, 19.2, 46.5, "Optical Stream")
    draw_arrow(ax, 36.2, 46.5, 37.4, 46.5, "Patient Folds")

    # Branching arrows from Stage 3 to Stage 4 Models
    draw_branching_arrows(ax, 57.4, 46.5, 59.5, [70.9, 43.9, 16.9], ["(1) Multi-Rate", "(2) 30s Context", "(3) Single-Rate"])

    # Converging arrows from Stage 4 Models to Stage 5 Decision
    draw_converging_arrows(ax, 82.2, [70.9, 43.9, 16.9], 84.3, 46.5)

    # Save to all target paths
    filename = "flow_osasud_ppg_apnea.png"
    for out_dir in OUTPUT_DIRS:
        dest_path = os.path.join(out_dir, filename)
        fig.savefig(dest_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Saved: {dest_path}")
    plt.close(fig)


# ==============================================================================
# 3. IMU FLOW DIAGRAM (SISFALL DATASET)
# ==============================================================================
def generate_imu_flow():
    fig, ax = create_base_canvas(
        title="END-TO-END PHYSIOLOGICAL PIPELINE: EMERGENCY FALL DETECTION VS ADL",
        subtitle="Modality: 6-Axis Inertial Measurement Unit (IMU) | Dataset: SisFall | Hardware: Waist MPU-9250 -> NVIDIA Jetson Orin Nano"
    )

    # 1. Stage 1: Acquisition (x: 1.5, w: 16.5)
    draw_stage_box(ax, 1.5, 3.5, 16.5, 86.0, 1, "Sensor & Signal", "Waist Wearable Hardware")
    s1_items = [
        {"type": "heading", "text": "Hardware Sensor:"},
        {"type": "kv", "k": "Sensor", "v": "6-Axis Waist Inertial Unit"},
        {"type": "kv", "k": "Reference ICs", "v": "ADXL345 Accel + ITG3200 Gyro"},
        {"type": "kv", "k": "Deployment IC", "v": "MPU-9250 6-Axis motion tracking"},
        {"type": "kv", "k": "Placement", "v": "Waist belt (center lumbar / hip)"},
        {"type": "divider"},
        {"type": "heading", "text": "Raw Signal Channels:"},
        {"type": "bullet", "text": "Tri-axial acceleration (ax, ay, az):"},
        {"type": "bullet", "text": "  • Calibrated units: g (32/8192 g/LSB)"},
        {"type": "bullet", "text": "  • Dynamic range: +/- 16 g"},
        {"type": "bullet", "text": "Tri-axial angular velocity (wx, wy, wz):"},
        {"type": "bullet", "text": "  • Units: deg/s (4000/65536 deg/s/LSB)"},
        {"type": "bullet", "text": "  • Dynamic range: +/- 2000 deg/s"},
        {"type": "kv", "k": "Sampling Rate", "v": "fs = 200 Hz native (100 Hz edge)"},
        {"type": "kv", "k": "Buffer Window", "v": "2.0 s streaming window (40 tokens)"},
        {"type": "kv", "k": "Hop Cadence", "v": "1.0 s sliding stride (50% overlap)"},
        {"type": "divider"},
        {"type": "heading", "text": "Hardware Interface:"},
        {"type": "bullet", "text": "I2C bus @ 400 kHz with continuous FIFO read"}
    ]
    render_content_lines(ax, 1.5, 84.5, 16.5, s1_items, max_width_chars=34, line_h=1.35)

    # 2. Stage 2: Dataset & Detection Target (x: 19.2, w: 17.0)
    draw_stage_box(ax, 19.2, 3.5, 17.0, 86.0, 2, "Dataset & Target", "Clinical Benchmark & Task")
    s2_items = [
        {"type": "heading", "text": "Public Benchmark Dataset:"},
        {"type": "kv", "k": "Dataset", "v": "SisFall Public Dataset (MDPI Sensors)"},
        {"type": "kv", "k": "Cohort Size", "v": "38 participants (23 young, 15 elderly)"},
        {"type": "kv", "k": "Total Trials", "v": "4,500 validated recording files"},
        {"type": "kv", "k": "Activity Scope", "v": "19 ADL types and 15 fall types"},
        {"type": "kv", "k": "Class Breakdown", "v": "2,702 ADL trials vs 1,798 falls"},
        {"type": "divider"},
        {"type": "heading", "text": "CRITICAL FIX: Duration Shortcut"},
        {"type": "bullet", "text": "Uncovered major literature flaw: fall trials were ~15 s, ADLs 10-180 s. A simple tree on duration gets 100% accuracy without sensor data!"},
        {"type": "bullet", "text": "OUR FIX: Strictly evaluated on fixed 2.0 s streaming unpadded windows."},
        {"type": "divider"},
        {"type": "heading", "text": "Patient-Independent Split:"},
        {"type": "bullet", "text": "5 subject-independent cross-validation folds"},
        {"type": "bullet", "text": "Zero subject overlap between train/dev/test"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Detection Target:"},
        {"type": "kv", "k": "Target", "v": "Acute Fall Impact vs High-Energy ADL"},
        {"type": "kv", "k": "Clinical Risks", "v": "Leading cause of accidental injury, hip fractures, and death in older adults"}
    ]
    render_content_lines(ax, 19.2, 84.5, 17.0, s2_items, max_width_chars=34, line_h=1.35)

    # 3. Stage 3: Preprocessing Pipeline: IPMV (x: 37.4, w: 20.0)
    draw_stage_box(ax, 37.4, 3.5, 20.0, 86.0, 3, "Preprocessing Pipeline", "IPMV: Impact-Preserving Multi-View")
    s3_items = [
        {"type": "heading", "text": "NOVELTY: Impact Preservation & Tilt Invariance"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 1: Native 200 Hz Impact Peak Path"},
        {"type": "bullet", "text": "Fall impact spikes last only 10-30 ms! Standard decimation completely smooths peaks."},
        {"type": "bullet", "text": "Tracks extrema and peak absolute derivative max|da/dt| at full 200 Hz before tokenization."},
        {"type": "divider"},
        {"type": "heading", "text": "Step 2: Adaptive Low-Pass Filter"},
        {"type": "bullet", "text": "30 Hz causal 2nd-order Butterworth with adaptive residual: passes more raw signal as acceleration norm surges to preserve true impact."},
        {"type": "divider"},
        {"type": "heading", "text": "Step 3: Dynamic Gravity Decoupling"},
        {"type": "bullet", "text": "0.75 Hz 2nd-order filter estimates 3D gravity g"},
        {"type": "bullet", "text": "Dynamic acceleration: a_dyn = a - g"},
        {"type": "bullet", "text": "Projects dynamic acceleration & rotation into vertical (parallel to g) and horizontal (perpendicular to g) components."},
        {"type": "bullet", "text": "RESULT: Mathematical invariance to belt sensor tilt and mounting rotation! (0 decision flips under 3D rotation)."},
        {"type": "divider"},
        {"type": "heading", "text": "Step 4: Causal Daubechies-2 Wavelet RMS"},
        {"type": "bullet", "text": "3-level decimated causal db2 wavelet decomposition"},
        {"type": "bullet", "text": "Tokens: 24 features every 50 ms -> 40 tokens (2 s)"}
    ]
    render_content_lines(ax, 37.4, 84.5, 20.0, s3_items, max_width_chars=39, line_h=1.35)

    # 4. Stage 4: Top Three Models (x: 58.6, w: 24.5)
    draw_stage_box(ax, 58.6, 3.5, 24.5, 86.0, 4, "Top Three Models", "Evaluated on Held-Out SisFall Participants")

    # Model 1 Card (y: 58.0, h: 25.8)
    m1_items = [
        {"type": "kv", "k": "Architecture", "v": "Compact Multi-Scale InceptionTime"},
        {"type": "bullet", "text": "6-module Inception-style temporal CNN (kernels 10, 20, 40 ms)"},
        {"type": "bullet", "text": "Residual bottleneck shortcuts + global average pooling"},
        {"type": "kv", "k": "Complexity", "v": "8,833 parameters (64.0 KB checkpoint)"},
        {"type": "heading", "text": "Held-Out Subject Test Results:"},
        {"type": "bullet", "text": "Subject-Balanced Accuracy: 98.48%  |  AUROC: 0.9991"},
        {"type": "bullet", "text": "Fall Recall: 98.22%  |  ADL Specificity: 98.70%"},
        {"type": "bullet", "text": "Fall F1-Score: 98.14%  |  False Alerts: 2.41/hour"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.82 ms forward pass on Jetson CPU"}
    ]
    draw_sub_card(ax, 59.5, 58.0, 22.7, 25.8, "1. Compact InceptionTime (inception)", m1_items, is_top=True, badge="NEURAL WINNER")

    # Model 2 Card (y: 31.0, h: 25.8)
    m2_items = [
        {"type": "kv", "k": "Architecture", "v": "100 Hz IPMV + Gradient Boosted Trees"},
        {"type": "bullet", "text": "60 shallow trees (depth 3) on 168 statistical & wavelet features"},
        {"type": "bullet", "text": "Exported as pure NumPy model (zero scikit-learn dependency!)"},
        {"type": "kv", "k": "Complexity", "v": "8,195 bytes pure NumPy runtime model"},
        {"type": "heading", "text": "Held-Out Subject Test Results:"},
        {"type": "bullet", "text": "Subject-Balanced Accuracy: 98.51%  |  AUROC: 0.9987"},
        {"type": "bullet", "text": "Fall Recall: 98.72% (highest fall sensitivity!)"},
        {"type": "bullet", "text": "ADL Specificity: 98.11%  |  Fall F1-Score: 98.04%"},
        {"type": "bullet", "text": "False Alerts: 3.59/hour (52 over 14.5 h ADLs)"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.53 ms forward (3.79 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 31.0, 22.7, 25.8, "2. 100 Hz IPMV + XGBoost (native_100)", m2_items, is_top=False, badge="EDGE PROTOTYPE")

    # Model 3 Card (y: 4.0, h: 25.8)
    m3_items = [
        {"type": "kv", "k": "Architecture", "v": "Teacher-Distilled Dual-Stream TCN (dm_distilled)"},
        {"type": "bullet", "text": "Dual-stream accel/gyro branches (kernels 3, 5, 9; dilations 1, 2, 4)"},
        {"type": "bullet", "text": "Learned sigmoid feature gating + temporal attention pooling"},
        {"type": "bullet", "text": "Trained via weak XGBoost witness soft targets (T=2)"},
        {"type": "kv", "k": "Complexity", "v": "5,846 parameters (41.6 KB ONNX graph)"},
        {"type": "heading", "text": "Held-Out Subject Test Results:"},
        {"type": "bullet", "text": "Subject-Balanced Accuracy: 98.61% mean (up to 99.04%)"},
        {"type": "bullet", "text": "Fall F1-Score: 98.88%  |  False Alerts: 2.21 - 3.10/h"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.73 ms forward (4.28 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 4.0, 22.7, 25.8, "3. Teacher-Distilled DM-TCN (dm_distilled)", m3_items, is_top=False, badge="DISTILLED DEEP")

    # 5. Stage 5: Output Decision & Alert Generation (x: 84.3, w: 14.2)
    draw_stage_box(ax, 84.3, 3.5, 14.2, 86.0, 5, "Decision & Alert", "Inference & Edge Output")
    s5_items = [
        {"type": "heading", "text": "Scoring Logic:"},
        {"type": "bullet", "text": "Max-window probability aggregation"},
        {"type": "bullet", "text": "Probability threshold tau = 0.50"},
        {"type": "divider"},
        {"type": "heading", "text": "Refractory Lockout:"},
        {"type": "bullet", "text": "5-second refractory lockout after alarm (prevents duplicate alerts during ground struggle)"},
        {"type": "bullet", "text": "Requires 2 consecutive below-threshold windows to re-arm detector"},
        {"type": "divider"},
        {"type": "heading", "text": "Emergency Actions:"},
        {"type": "bullet", "text": "State: FALL_EMERGENCY"},
        {"type": "bullet", "text": "Immediate BLE / Wi-Fi caregiver notification"},
        {"type": "bullet", "text": "Local buzzer activation with 30 s user cancel button"},
        {"type": "divider"},
        {"type": "heading", "text": "Edge Budget (Jetson):"},
        {"type": "kv", "k": "E2E Latency", "v": "3.79 - 4.78 ms"},
        {"type": "kv", "k": "Hop Deadline", "v": "1,000 ms (1 s hop)"},
        {"type": "kv", "k": "Budget Consumed", "v": "0.48% - 0.71%"},
        {"type": "kv", "k": "Deadline Status", "v": "REAL-TIME PASS"}
    ]
    render_content_lines(ax, 84.3, 84.5, 14.2, s5_items, max_width_chars=27, line_h=1.35)

    # Connecting Arrows
    draw_arrow(ax, 18.0, 46.5, 19.2, 46.5, "6-Axis Stream")
    draw_arrow(ax, 36.2, 46.5, 37.4, 46.5, "Fixed 2s Windows")

    # Branching arrows from Stage 3 to Stage 4 Models
    draw_branching_arrows(ax, 57.4, 46.5, 59.5, [70.9, 43.9, 16.9], ["(1) 40 Tokens", "(2) 168 Descriptors", "(3) Dual Stream"])

    # Converging arrows from Stage 4 Models to Stage 5 Decision
    draw_converging_arrows(ax, 82.2, [70.9, 43.9, 16.9], 84.3, 46.5)

    # Save to all target paths
    filename = "flow_sisfall_imu_fall.png"
    for out_dir in OUTPUT_DIRS:
        dest_path = os.path.join(out_dir, filename)
        fig.savefig(dest_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Saved: {dest_path}")
    plt.close(fig)


# ==============================================================================
# 4. EDA FLOW DIAGRAM (WESAD DATASET)
# ==============================================================================
def generate_eda_flow():
    fig, ax = create_base_canvas(
        title="END-TO-END PHYSIOLOGICAL PIPELINE: AUTONOMIC STRESS & SYMPATHETIC AROUSAL",
        subtitle="Modality: Electrodermal Activity (EDA/GSR) + Motion | Dataset: WESAD | Hardware: Wristband Wearable -> NVIDIA Jetson Orin Nano"
    )

    # 1. Stage 1: Acquisition (x: 1.5, w: 16.5)
    draw_stage_box(ax, 1.5, 3.5, 16.5, 86.0, 1, "Sensor & Signal", "Wrist Wearable Hardware")
    s1_items = [
        {"type": "heading", "text": "Hardware Sensor:"},
        {"type": "kv", "k": "Sensor", "v": "Wristband Wearable (Empatica E4 / Grove GSR)"},
        {"type": "kv", "k": "Electrodes", "v": "Ag/AgCl dry contact snap electrodes"},
        {"type": "kv", "k": "Placement", "v": "Ventral wrist surface (sweat gland dense)"},
        {"type": "divider"},
        {"type": "heading", "text": "Signal Specifications:"},
        {"type": "kv", "k": "Primary Channel", "v": "Electrodermal Activity (EDA / GSR)"},
        {"type": "kv", "k": "Physical Units", "v": "Skin conductance in microSiemens (uS)"},
        {"type": "kv", "k": "EDA Rate", "v": "fs = 4 Hz continuous streaming"},
        {"type": "kv", "k": "Motion Channel", "v": "3-Axis Wrist Accelerometer (ax, ay, az)"},
        {"type": "kv", "k": "Motion Rate", "v": "fs = 32 Hz (in g units)"},
        {"type": "kv", "k": "Buffer Window", "v": "60-second sliding analysis window"},
        {"type": "kv", "k": "Hop Stride", "v": "15-second hop (75% overlap)"},
        {"type": "divider"},
        {"type": "heading", "text": "Resting Calibration:"},
        {"type": "bullet", "text": "Single 120-second resting baseline capture recorded per subject before active monitoring"}
    ]
    render_content_lines(ax, 1.5, 84.5, 16.5, s1_items, max_width_chars=34, line_h=1.35)

    # 2. Stage 2: Dataset & Detection Target (x: 19.2, w: 17.0)
    draw_stage_box(ax, 19.2, 3.5, 17.0, 86.0, 2, "Dataset & Target", "Clinical Benchmark & Task")
    s2_items = [
        {"type": "heading", "text": "Public Benchmark Dataset:"},
        {"type": "kv", "k": "Dataset", "v": "WESAD Public Benchmark (ACM ICMI)"},
        {"type": "kv", "k": "Cohort Size", "v": "15 healthy human participants"},
        {"type": "kv", "k": "Total Windows", "v": "1,008 baseline & 628 acute stress windows"},
        {"type": "kv", "k": "Stress Protocol", "v": "Trier Social Stress Test (TSST) + mental arithmetic"},
        {"type": "divider"},
        {"type": "heading", "text": "Evaluation Protocol:"},
        {"type": "bullet", "text": "15-fold Leave-One-Subject-Out (LOSO):"},
        {"type": "bullet", "text": "  • 14 subjects: Model training & feature scaling"},
        {"type": "bullet", "text": "  • 1 untouched subject: Pure out-of-fold test"},
        {"type": "bullet", "text": "Repeated across all 15 subjects (equal weights)"},
        {"type": "bullet", "text": "Zero subject data leak between train and test!"},
        {"type": "divider"},
        {"type": "heading", "text": "Clinical Detection Target:"},
        {"type": "kv", "k": "Condition", "v": "Sympathetic Nervous Autonomic Arousal"},
        {"type": "kv", "k": "Physiology", "v": "Sympathetic innervation of palmar eccrine sweat glands during acute stress"},
        {"type": "kv", "k": "Clinical Risks", "v": "Hypertension exacerbation, cognitive distress, nocturnal sleep disruption in elderly"}
    ]
    render_content_lines(ax, 19.2, 84.5, 17.0, s2_items, max_width_chars=34, line_h=1.35)

    # 3. Stage 3: Preprocessing Pipeline: MC-SAED (x: 37.4, w: 20.0)
    draw_stage_box(ax, 37.4, 3.5, 20.0, 86.0, 3, "Preprocessing Pipeline", "MC-SAED: Causal Quality-Weighted DSP")
    s3_items = [
        {"type": "heading", "text": "NOVELTY: Causal Real-Time Decomposition"},
        {"type": "bullet", "text": "Standard clinical cvxEDA requires non-causal convex optimization across the entire session (~8,290 ms). MC-SAED is fully causal, stateful, and executes in 0.26 ms with zero data leakage!"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 1: Contact Quality & Motion Gating"},
        {"type": "bullet", "text": "Validates physical range: 0 <= conductance <= 100 uS"},
        {"type": "bullet", "text": "Tracks rolling derivative MAD; flags flatline / step drops"},
        {"type": "bullet", "text": "Sample quality score (0.0 to 1.0) conditions tracker"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 2: Asymmetric Tonic / Phasic Separation"},
        {"type": "bullet", "text": "Skin conductance = Tonic (drift) + Phasic (spikes)"},
        {"type": "bullet", "text": "Asymmetric tracker: decay_rate > rise_rate"},
        {"type": "bullet", "text": "Relaxes downwards quickly when sweat reabsorbs, but follows upward surges very slowly. Prevents acute SCR peaks from pulling up the baseline."},
        {"type": "bullet", "text": "Rectified positive excursion gives pure Phasic SCR"},
        {"type": "divider"},
        {"type": "heading", "text": "Step 3: Personalized Resting Calibration"},
        {"type": "bullet", "text": "Normalizes components against 2-min resting capture:"},
        {"type": "bullet", "text": "  rel_tonic = (tonic - median) / (1.4826 * MAD)"},
        {"type": "bullet", "text": "Eliminates 500% inter-subject baseline conductance spread without leaking future data (+0.095 F1 gain!)"}
    ]
    render_content_lines(ax, 37.4, 84.5, 20.0, s3_items, max_width_chars=39, line_h=1.35)

    # 4. Stage 4: Top Three Models (x: 58.6, w: 24.5)
    draw_stage_box(ax, 58.6, 3.5, 24.5, 86.0, 4, "Top Three Models", "Evaluated on Held-Out LOSO Subjects")

    # Model 1 Card (y: 58.0, h: 25.8)
    m1_items = [
        {"type": "kv", "k": "Architecture", "v": "31-Coefficient Logistic Regression Model"},
        {"type": "bullet", "text": "Operates on 31 extracted MC-SAED tonic/phasic and wrist motion features across a 60 s window"},
        {"type": "bullet", "text": "Exported as portable JSON linear model (zero DL dependency!)"},
        {"type": "kv", "k": "Complexity", "v": "31 parameters (3,607 bytes JSON export)"},
        {"type": "heading", "text": "Held-Out LOSO Test Results:"},
        {"type": "bullet", "text": "Macro F1-Score: 0.873 +/- 0.117  |  Accuracy: 88.6%"},
        {"type": "bullet", "text": "Stress Sensitivity: 82.9%  |  Baseline Specificity: 92.4%"},
        {"type": "bullet", "text": "AUROC: 0.966  |  AUPRC: 0.954  |  Brier Score: 0.112"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.11 ms forward (14.38 ms end-to-end!)"}
    ]
    draw_sub_card(ax, 59.5, 58.0, 22.7, 25.8, "1. Calibrated Logistic (wesad_loso_v1)", m1_items, is_top=True, badge="DEPLOYMENT WINNER")

    # Model 2 Card (y: 31.0, h: 25.8)
    m2_items = [
        {"type": "kv", "k": "Architecture", "v": "Reliability-Aware Rest-Calibrated Fusion"},
        {"type": "bullet", "text": "Multi-branch neural fusion network"},
        {"type": "bullet", "text": "Combines rest-calibrated tonic & phasic representations"},
        {"type": "bullet", "text": "Active sensor contact reliability gate modulates branch weights"},
        {"type": "kv", "k": "Complexity", "v": "8,275 trainable parameters (46.2 KB ONNX)"},
        {"type": "heading", "text": "Held-Out LOSO Test Results:"},
        {"type": "bullet", "text": "Macro F1-Score: 0.899 +/- 0.130 (Top Multimodal!)"},
        {"type": "bullet", "text": "Overall Accuracy: 90.9%  |  Balanced Accuracy: 90.5%"},
        {"type": "bullet", "text": "Stress Sensitivity: 87.9%  |  Baseline Specificity: 93.0%"},
        {"type": "bullet", "text": "AUROC: 0.986  |  AUPRC: 0.979"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.78 ms forward pass on Jetson CPU"}
    ]
    draw_sub_card(ax, 59.5, 31.0, 22.7, 25.8, "2. Reliability Fusion (reliable_rest)", m2_items, is_top=False, badge="CATEGORY LEADER")

    # Model 3 Card (y: 4.0, h: 25.8)
    m3_items = [
        {"type": "kv", "k": "Architecture", "v": "Physics-Guided Multiscale Network (PG-MSTN)"},
        {"type": "bullet", "text": "Dilated Tonic TCN (captures slow baseline vasomotor drift)"},
        {"type": "bullet", "text": "Multiscale Phasic TCN (parallel receptive fields for fast SCR)"},
        {"type": "bullet", "text": "Soft motion quality gate suppresses artifact surges"},
        {"type": "kv", "k": "Complexity", "v": "9,323 trainable parameters (52.1 KB ONNX)"},
        {"type": "heading", "text": "Held-Out LOSO Test Results:"},
        {"type": "bullet", "text": "Macro F1-Score: 0.841 (zero-cal) / 0.887 (rest-cal)"},
        {"type": "bullet", "text": "AUROC: 0.968  |  Specificity: 89.3% - 95.1%"},
        {"type": "bullet", "text": "Sensitivity: 80.6% - 84.2%"},
        {"type": "kv", "k": "Jetson Orin Nano", "v": "0.85 ms forward (30.70 ms end-to-end)"}
    ]
    draw_sub_card(ax, 59.5, 4.0, 22.7, 25.8, "3. Physics-Guided TCN (pg_mstn)", m3_items, is_top=False, badge="PURE NEURAL")

    # 5. Stage 5: Output Decision & Alert Generation (x: 84.3, w: 14.2)
    draw_stage_box(ax, 84.3, 3.5, 14.2, 86.0, 5, "Decision & Alert", "Inference & Edge Output")
    s5_items = [
        {"type": "heading", "text": "Probability Scoring:"},
        {"type": "bullet", "text": "LOSO Platt scaling yields calibrated stress probability p"},
        {"type": "bullet", "text": "Decision threshold: tau = 0.50"},
        {"type": "divider"},
        {"type": "heading", "text": "Motion Artifact Filter:"},
        {"type": "bullet", "text": "If wrist accelerometry energy > threshold, flag PHYSICAL_EXERTION to prevent false sympathetic calls"},
        {"type": "bullet", "text": "Trailing window smoothing"},
        {"type": "divider"},
        {"type": "heading", "text": "Autonomic Outputs:"},
        {"type": "bullet", "text": "State: STRESS_AROUSAL vs CALM_BASELINE"},
        {"type": "bullet", "text": "Real-time SCR frequency/min"},
        {"type": "bullet", "text": "Longitudinal circadian log"},
        {"type": "divider"},
        {"type": "heading", "text": "Edge Budget (Jetson):"},
        {"type": "kv", "k": "E2E Latency", "v": "14.38 - 32.66 ms"},
        {"type": "kv", "k": "Stride Deadline", "v": "15,000 ms (15 s stride)"},
        {"type": "kv", "k": "Budget Consumed", "v": "0.10% - 0.22%"},
        {"type": "kv", "k": "Deadline Status", "v": "REAL-TIME PASS"}
    ]
    render_content_lines(ax, 84.3, 84.5, 14.2, s5_items, max_width_chars=27, line_h=1.35)

    # Connecting Arrows
    draw_arrow(ax, 18.0, 46.5, 19.2, 46.5, "EDA+ACC Stream")
    draw_arrow(ax, 36.2, 46.5, 37.4, 46.5, "60s Windows")

    # Branching arrows from Stage 3 to Stage 4 Models
    draw_branching_arrows(ax, 57.4, 46.5, 59.5, [70.9, 43.9, 16.9], ["(1) 31 Descriptors", "(2) Rest-Cal Fusion", "(3) Tonic/Phasic"])

    # Converging arrows from Stage 4 Models to Stage 5 Decision
    draw_converging_arrows(ax, 82.2, [70.9, 43.9, 16.9], 84.3, 46.5)

    # Save to all target paths
    filename = "flow_wesad_eda_stress.png"
    for out_dir in OUTPUT_DIRS:
        dest_path = os.path.join(out_dir, filename)
        fig.savefig(dest_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Saved: {dest_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Flow 1: ECG (PTB-XL)...")
    generate_ecg_flow()
    print("Generating Flow 2: PPG (OSASUD)...")
    generate_ppg_flow()
    print("Generating Flow 3: IMU (SisFall)...")
    generate_imu_flow()
    print("Generating Flow 4: EDA (WESAD)...")
    generate_eda_flow()
    print("\nAll four flow diagrams generated successfully in pure black-and-white publication quality!")
