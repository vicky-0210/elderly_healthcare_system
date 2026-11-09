"""
Script to generate four high-resolution, black-and-white publication-quality
flow diagrams for the top three models of each physiological dataset:
1. PTB-XL (ECG) - Left Ventricular Hypertrophy (LVH) Detection
2. OSASUD (PPG) - Sleep Apnea & Hypopnea Detection
3. SisFall (IMU) - Fall Detection vs Activities of Daily Living
4. WESAD (EDA) - Stress & Sympathetic Arousal Detection
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Ensure directories exist
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

    ax.text(50, 96.0, title, color="#FFFFFF", fontsize=17, fontweight="bold", ha="center", va="center", family="sans-serif")
    ax.text(50, 93.2, subtitle, color="#E0E0E0", fontsize=11, fontweight="normal", ha="center", va="center", family="sans-serif")

    # Footer note
    ax.text(50, 0.8, "Elderly Healthcare Edge System | NVIDIA Jetson Orin Nano Deployment | Strictly Patient-Independent Evaluation | Pure B&W Flowchart",
            color="#555555", fontsize=9, ha="center", va="center", style="italic")

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
    pill_h = 4.2
    pill = FancyBboxPatch(
        (x, y + h - pill_h), w, pill_h,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        facecolor="#000000", edgecolor="#000000", linewidth=1.0
    )
    ax.add_patch(pill)

    header_text = f"STAGE {stage_num}: {title.upper()}"
    ax.text(x + w / 2.0, y + h - pill_h / 2.0 + (0.5 if subtitle else 0.0), header_text,
            color="#FFFFFF", fontsize=10.5, fontweight="bold", ha="center", va="center")
    if subtitle:
        ax.text(x + w / 2.0, y + h - pill_h / 2.0 - 1.2, subtitle,
                color="#CCCCCC", fontsize=8.0, ha="center", va="center")

def draw_sub_card(ax, x, y, w, h, title, lines, is_top=False, badge=""):
    """Draws an inner structured card with clear typography."""
    card_bg = "#FFFFFF" if not is_top else "#FAFAFA"
    card_edge = "#000000"
    card_lw = 1.8 if is_top else 1.2
    card_ls = "-" if is_top else "--"

    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor=card_bg, edgecolor=card_edge, linewidth=card_lw, linestyle=card_ls
    )
    ax.add_patch(card)

    # Title bar inside sub-card
    title_bar_h = 2.4
    tb = FancyBboxPatch(
        (x, y + h - title_bar_h), w, title_bar_h,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor="#222222" if is_top else "#E0E0E0",
        edgecolor=card_edge, linewidth=1.0
    )
    ax.add_patch(tb)

    t_color = "#FFFFFF" if is_top else "#000000"
    ax.text(x + 0.8, y + h - title_bar_h / 2.0, title,
            color=t_color, fontsize=9.2, fontweight="bold", ha="left", va="center")

    if badge:
        ax.text(x + w - 0.8, y + h - title_bar_h / 2.0, badge,
                color=t_color, fontsize=7.5, fontweight="bold", ha="right", va="center")

    # Render lines
    cur_y = y + h - title_bar_h - 1.0
    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            # Subheading inside card
            clean_text = line.replace("**", "")
            ax.text(x + 0.8, cur_y, clean_text, color="#000000", fontsize=8.2, fontweight="bold", ha="left", va="top")
            cur_y -= 1.6
        elif line.startswith("- "):
            # Bullet point
            clean_text = line[2:]
            ax.text(x + 0.8, cur_y, "•", color="#000000", fontsize=8.0, fontweight="bold", ha="left", va="top")
            ax.text(x + 1.8, cur_y, clean_text, color="#111111", fontsize=7.7, fontweight="normal", ha="left", va="top")
            cur_y -= 1.55
        elif line == "":
            cur_y -= 0.6
        else:
            ax.text(x + 0.8, cur_y, line, color="#222222", fontsize=7.7, ha="left", va="top")
            cur_y -= 1.55

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """Draws a crisp black connecting arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5,head_width=3.5",
        color="#000000", linewidth=1.6, mutation_scale=1.0
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0 + 1.0
        ax.text(mid_x, mid_y, label, color="#000000", fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor="#000000", linewidth=0.8))

print("Base layout helper defined.")
