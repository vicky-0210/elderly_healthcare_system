"""
build_reorganized_report.py
Generates the reorganized, professional 40-page B.Tech Project Report:
- Pages 1-5: Exactly untouched (Title, Approval, Certificate, Declaration, Acknowledgements)
- Pages 6-11: Front matter (Abstract, Contents, List of Figures, List of Tables, List of Abbreviations)
- Chapters 1-5: Highly detailed, comprehensive technical chapters with visual results placed in their relevant sections
- Includes Jetson Orin Nano edge deployment results (Figure 1-4, confusion matrices, latency tables)
- Excludes redundant/noisy visual results
- Total pages: exactly 40 pages
Outputs:
- Wearable_Health_Monitoring_Progress.docx
- Wearable_Health_Monitoring_Progress.pdf
"""

from pathlib import Path
import os, sys, re, csv, hashlib
from PIL import Image, ImageOps
from pypdf import PdfReader
from docx import Document
from docx.shared import Pt, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from xml.sax.saxutils import escape

# Paths
repo = Path(r"C:\Final Project\elderly_healthcare_system")
base = Path(r"C:\Final Project\LVH_Detection")
out_dirs = [
    repo / "research" / "thesis",
    base / "output" / "thesis",
    Path(r"C:\Final Project")
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)

work = base / "tmp" / "thesis_render"
work.mkdir(parents=True, exist_ok=True)
results = repo / "research" / "results"
jetson_dir = repo / "jetson_results"
thesis_figs = repo / "research" / "thesis" / "figures"

# Helper for CSV
def rows(rel_path):
    p = results / rel_path
    if not p.exists():
        p = repo / rel_path
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))

def n(x):
    return f"{float(x):.3f}"

# Page structure
pages = []
group = None

def page(title, section="", front=False):
    global group
    group = []
    pages.append([title, section, group, front])

def add(kind, *args):
    group.append((kind, *args))

def p(text):
    add("p", text)

def h(text):
    add("h", text)

def c(text):
    add("c", text)

def gap(sz):
    add("gap", sz)

def table(title, headers, rows_data, widths):
    add("table", title, headers, rows_data, widths)

def fig(path, caption, max_h=190):
    add("fig", path, caption, max_h)

def ref(text):
    add("ref", text)

# -------------------------------------------------------------------------
# BUILD PAGES
# -------------------------------------------------------------------------

title_text = "Multimodal Wearable Health Monitoring for Older Adults"
team_members = [
    "Vivek Kumar  -  23ECB0A09",
    "Bhanu Pratap  -  23ECB0A27",
    "Ujwal Kumar  -  23ECB0A16"
]

# --- PAGE 1: TITLE PAGE (Front) ---
page("", front=True)
gap(20)
add("title", title_text)
gap(16)
c("Project report submitted to the")
c("National Institute of Technology Warangal")
c("For award of the degree")
c("Of")
c("Bachelor of Technology")
c("in Electronics and Communication Engineering")
gap(14)
c("by")
for m in team_members:
    c(m)
gap(12)
c("Under the guidance of")
c("Prof. Mohammad Farukh Hashmi")
gap(14)
logo_path = work / "institute_emblem.png"
if not logo_path.exists():
    template_pdf = PdfReader(r"C:\Users\vivek\Downloads\Template_for_Thesis_2024-10-1-17-14-35.pdf")
    template_pdf.pages[0].images[0].image.convert("L").save(logo_path)
add("logo", logo_path)
gap(14)
c("Department of Electronics and Communication Engineering")
c("National Institute of Technology Warangal")
c("November 2026")

# --- PAGE 2: APPROVAL OF THE VIVA-VOCE BOARD (Front) ---
page("APPROVAL OF THE VIVA-VOCE BOARD", front=True)
c("09/11/26")
gap(10)
p("Certified that the project report entitled Multimodal Health Monitoring for Older Adults submitted by Vivek Kumar(23ECB0A09), Bhanu Pratap(23ECB0A27), Ujwal Kumar(23ECB0A016) to the National Institute of Technology Warangal, for the award of the degree of Bachelor of Technology, has been accepted by the external examiners, and that the students have successfully defended the report in the viva-voce examination held today.")
gap(35)
table("", ["", "", ""], [
    ["(Member, Evaluation Committee)", "(Member, Evaluation Committee)", "(Member, Evaluation Committee)"],
    ["", "", ""],
    ["(Supervisor)", "(Co-Supervisor)", ""],
    ["", "", ""],
    ["(External Examiner)", "(Head of the Department)", ""]
], [0.33, 0.33, 0.34])

# --- PAGE 3: CERTIFICATE (Front) ---
page("CERTIFICATE", front=True)
p("This is to certify that the project report entitled Multimodal Health Monitoring for Older Adults submitted by Vivek Kumar(23ECB0A09), Bhanu Pratap(23ECB0A27), Ujwal Kumar(23ECB0A016) to the National Institute of Technology Warangal, is a record of bona fide work carried out under my (our) supervision, and we consider it worthy of consideration for the award of the degree of Bachelor of Technology of the Institute.")
gap(65)
table("", ["", ""], [
    ["_____________________________\nProf. Ravi Kumar Jatoth\nHead of the Department\nDepartment of ECE\nNIT Warangal",
     "_____________________________\nProf. Mohammad Farukh Hashmi\nAssistant Professor\nDepartment of ECE\nNIT Warangal"]
], [0.5, 0.5])

# --- PAGE 4: DECLARATION (Front) ---
page("DECLARATION", front=True)
p("We certify that")
p("a.  The work contained in this report is original and has been done by us under the general supervision of our supervisor(s).")
p("b.  The work has not been submitted to any other Institute for any degree or diploma.")
p("c.  We have followed the guidelines provided by the Institute in writing the report.")
p("d.  Whenever we have used materials (data, theoretical analysis and text) from other sources, we have given due credit to them by citing them in the text of the report and giving their details in the references.")
p("e.  Whenever we have quoted written materials from other sources, we have put them under quotation marks and given due credit to the sources by citing them and giving the required details in the references.")
gap(40)
p("Signature: ____________________      Vivek Kumar(23ECB0A09)")
gap(10)
p("Signature: ____________________      Bhanu Pratap(23ECB0A27)")
gap(10)
p("Signature: ____________________      Ujwal Kumar(23ECB0A16)")
gap(25)
p("Date: 09/11/2026")

# --- PAGE 5: ACKNOWLEDGEMENTS (Front) ---
page("Acknowledgements", front=True)
p("We thank our guide, Prof. Mohammad Farukh Hashmi, for steady guidance through a project that grew well beyond its first proposal, and for pushing us to test our own results as hard as anyone else's. We thank Prof. J. Ravi Kumar, the Head of the Department, and the faculty of the Department of Electronics and Communication Engineering for their support, reviews, and constructive feedback throughout the year.")
p("We acknowledge the participants, clinicians, and research teams who made PTB-XL, OSASUD, SisFall, and WESAD publicly available. Their rigorous clinical recordings and expert annotations made it possible to study four fundamental physiological sensing modalities and benchmark edge-deployable deep learning pipelines prior to collecting prospective physical wearable measurements.")
p("We also thank our colleagues, lab staff, and families whose encouragement made the completion of this undergraduate research project possible.")
gap(35)
c("Vivek Kumar (23ECB0A09)")
c("Bhanu Pratap (23ECB0A27)")
c("Ujwal Kumar (23ECB0A16)")

# --- PAGE 6: ABSTRACT (Front) ---
page("Abstract", front=True)
p("We investigated how far a wearable health monitoring system for older adults can progress using public physiological recordings before collecting its own physical hardware measurements. Four independent, decoupled pipelines estimate Left Ventricular Hypertrophy (LVH) associated ECG patterns, apnea or hypopnea from optical pulse photoplethysmography (PPG), simulated falls from waist inertial measurement units (IMU), and autonomic stress-associated arousal from wrist electrodermal activity (EDA). Each physiological modality operates with an independent causal sampling clock, physical-voltage or acceleration preservation, signal-quality gating, and participant-independent cross-validation.")
p("For PTB-XL single-lead Lead II ECG, a multiscale residual convolutional neural network (CNN) achieved an AUROC of 0.9329 and AUPRC of 0.8838 under strict normal controls, while a privileged 12-lead Inception teacher was distilled into an edge student achieving 0.437 patient average precision (AP) across broad clinical rest records. For OSASUD optical PPG, a dual-rate temporal convolutional network (TCN) with a GRU matched respiratory events at 0.247 patient AP, reducing false alarms to 56.2 per hour (a 17.7/h reduction over single-rate baselines). For SisFall waist IMU, an Impact-Preserving Multi-View (IPMV) pipeline eliminated the 100% file-duration leak prevalent in literature, achieving 98.51% subject-balanced accuracy with a compact 8,195-byte NumPy XGBoost model and 98.61% mean accuracy with a teacher-distilled dual-stream TCN. For WESAD wrist EDA, a causal asymmetric exponential decomposition (MC-SAED) executed in 0.26 ms/min; when coupled with a 2-minute resting baseline calibration, a lightweight 31-coefficient logistic model achieved 0.873 macro F1, and a reliability-aware neural fusion model achieved 0.899 macro F1.")
p("Hardware benchmarks on an NVIDIA Jetson Orin Nano edge board verified that end-to-end processing across all four pipelines consumes less than 2.0% of their respective real-time sampling deadlines (aggregate compute < 5% of one Cortex-A78AE core), demonstrating real-time edge feasibility. These findings establish that disciplined causal DSP, coordinate invariance, and participant-independent evaluation are essential prerequisites before prospective geriatric clinical deployment.")
gap(10)
p("Index terms — Wearable sensors, edge computing, NVIDIA Jetson Orin Nano, electrocardiography (ECG), photoplethysmography (PPG), inertial measurement unit (IMU), electrodermal activity (EDA), subject-independent evaluation.")

# --- PAGE 7: CONTENTS (Front) ---
page("Contents", front=True)
add("toc")

# --- PAGE 8: LIST OF FIGURES (Front) ---
page("List of Figures", front=True)
add("figlist", 0)

# --- PAGE 9: LIST OF FIGURES (CONTINUED) (Front) ---
page("List of Figures (continued)", front=True)
add("figlist", 1)

# --- PAGE 10: LIST OF TABLES (Front) ---
page("List of Tables", front=True)
add("tableslist")

# --- PAGE 11: LIST OF ABBREVIATIONS (Front) ---
page("List of Abbreviations", front=True)
abbrs = [
    ("ADC", "Analog-to-digital converter"),
    ("ADL", "Activity of daily living"),
    ("AP", "Average precision (PR curve)"),
    ("AUROC", "Area under ROC curve"),
    ("BA", "Balanced accuracy"),
    ("BVP", "Blood volume pulse"),
    ("CNN", "Convolutional neural network"),
    ("cvxEDA", "Convex optimization EDA decomp."),
    ("DSP", "Digital signal processing"),
    ("ECG", "Electrocardiogram"),
    ("EDA / GSR", "Electrodermal activity / GSR"),
    ("FIFO", "First-in first-out buffer"),
    ("FIR", "Finite impulse response filter"),
    ("GRU", "Gated recurrent unit"),
    ("IBI", "Interbeat interval"),
    ("IMU", "Inertial measurement unit"),
    ("IPMV", "Impact-preserving multi-view prep."),
    ("IQR / MAD", "Interquartile range / MAD"),
    ("KD", "Knowledge distillation"),
    ("LOSO", "Leave-one-subject-out CV"),
    ("LPF", "Low-pass filter"),
    ("LVH", "Left ventricular hypertrophy"),
    ("MC-SAED", "Motion-cond. sub-harm. asym. exp."),
    ("MIL", "Multiple instance learning"),
    ("ONNX", "Open Neural Network Exchange"),
    ("OSA", "Obstructive sleep apnea"),
    ("PPG", "Photoplethysmography"),
    ("PSG", "Polysomnography"),
    ("RMS", "Root mean square"),
    ("SCL", "Skin conductance level (tonic)"),
    ("SCR", "Skin conductance response (phasic)"),
    ("SSM", "State-space model"),
    ("TCN", "Temporal convolutional network"),
    ("TSST", "Trier Social Stress Test"),
    ("XGBoost", "Extreme Gradient Boosting")
]
half = (len(abbrs) + 1) // 2
abbr_rows = []
for i in range(half):
    k1, v1 = abbrs[i]
    k2, v2 = abbrs[i + half] if (i + half) < len(abbrs) else ("", "")
    abbr_rows.append([k1, v1, k2, v2])

table("", ["", "", "", ""], abbr_rows, [0.14, 0.36, 0.14, 0.36])

# --- PAGE 12 (1): CHAPTER 1 - INTRODUCTION (Part 1) ---
page("Introduction", "Chapter 1")
h("1.1   Measurement objectives and geriatric care")
p("Continuous wearable monitoring holds immense promise for elderly healthcare, where undetected physiological deterioration frequently leads to preventable hospitalizations, catastrophic falls, and cardiovascular emergencies. However, older adults exhibit unique physiological characteristics that challenge conventional consumer wearables: diminished peripheral perfusion, chronic cardiac arrhythmias, gait instability, and altered skin conductance dynamics. A waist sensor can register an acute impact within 20 milliseconds, while nocturnal respiratory hypopneas unfold over 10 to 60 seconds, and autonomic stress surges manifest across minutes.")
p("Our proposed system enforces an asynchronous, decoupled architecture where four vital physiological signals are monitored independently on an edge board. A sensor disconnect or motion artifact in one stream (e.g. wrist EDA) never invalidates or halts an emergency alert in another stream (e.g. waist fall detection).")
gap(6)
table("1.1  Wearable inputs, sensors, and study endpoints",
      ["Module", "Sensor", "Source rate", "Target edge stream", "Research endpoint"],
      [
          ["ECG", "BioAmp EXG Pill", "500 Hz", "360 Hz (mV)", "LVH-associated Lead II cardiac pattern"],
          ["PPG", "MAX30102 Optical", "80/100 Hz", "31 feats/s", "Apnea / hypopnea respiratory intervals"],
          ["IMU", "Waist MPU-9250", "200 Hz", "100 Hz (g, deg/s)", "Acute impact falls vs daily activities"],
          ["EDA", "Wrist GSR / E4", "4 Hz", "4 Hz (uS) + 32 Hz ACC", "Autonomic sympathetic stress vs baseline"]
      ],
      [0.12, 0.24, 0.16, 0.20, 0.28])

# --- PAGE 13 (2): CHAPTER 1 - INTRODUCTION (Part 2) ---
page("Introduction (continued)", "Chapter 1")
h("1.2   Target edge computing platform: NVIDIA Jetson Orin Nano")
p("Edge deployment requires running complex machine learning pipelines locally on battery-operated embedded hardware without offloading raw physiological waveforms to remote cloud servers. This project targets the NVIDIA Jetson Orin Nano Developer Kit, featuring a 6-core ARM Cortex-A78AE v8.2 64-bit CPU @ 1.5 GHz and an NVIDIA Ampere architecture GPU with 1,024 CUDA cores and 32 Tensor Cores.")
p("The computational challenge lies in executing causal FIR filtering, wavelet decompositions, feature extraction, and neural/classical forward passes within stringent real-time hop deadlines (ranging from 1.0 s for IMU to 15.0 s for EDA). The target benchmark requires that each module consumes less than 2.0% of its allotted hop time, ensuring that aggregate system compute leaves over 95% of processing capacity free for hardware I2C acquisition, database logging, and wireless alert transmission.")
h("1.3   Scope of the progress review")
p("This report presents a comprehensive academic investigation into the dataset audits, mathematical signal processing foundations, model architectures, rigorous participant-independent evaluation, and hardware deployment on the Jetson Orin Nano. The local records preserve the public clinical datasets, enabling exact reproduction of every fitted parameter, ROC curve, and confusion matrix.")

# --- PAGE 14 (3): CHAPTER 2 - LITERATURE REVIEW (Part 1) ---
page("Literature Review", "Chapter 2")
h("2.1   The reference defines the learning task")
p("Wagner et al. [1] established the PTB-XL corpus as the primary benchmark for deep-learning ECG interpretation. Standard clinical criteria for Left Ventricular Hypertrophy (LVH), such as the Sokolow-Lyon and Cornell indices, rely on 12-lead voltage sums (e.g. S in V1 + R in V5/V6 >= 3.5 mV). Wearable single-lead approximations (Lead II) forfeit spatial cross-lead context, raising the fundamental question: can a single physical-voltage Lead II channel reliably capture LVH patterns when trained via privileged multi-lead knowledge distillation?")
p("In nocturnal respiratory monitoring, the OSASUD database [3] pairs bedside optical plethysmography with gold-standard polysomnography (PSG) in stroke-unit patients. Standard methods in literature apply whole-window sliding classifications (e.g. 30 s blocks) which blur the exact second-level event onset boundaries. Furthermore, aggressive bandpass filtering discards baseline drift, discarding low-frequency respiratory modulations.")
h("2.2   Critical protocol flaws and literature shortcuts")
p("In inertial fall detection, Sucerquia et al. [4] provided the SisFall benchmark. A critical literature flaw uncovered during our research audit is that SisFall fall trials are roughly 15 seconds in duration, whereas ADLs span 10 to 180 seconds. A simple 3-node decision tree trained solely on recording duration achieves 100% out-of-fold accuracy without inspecting any sensor data! Many published papers suffer from this fatal data leak. A rigorous pipeline must train strictly on fixed, unpadded streaming windows.")

# --- PAGE 15 (4): CHAPTER 2 - LITERATURE REVIEW (Part 2) ---
page("Literature Review (continued)", "Chapter 2")
h("2.3   Edge intelligence and causal signal decomposition")
p("In electrodermal activity analysis, Schmidt et al. [5] introduced WESAD to study multimodal affect. Classical clinical decomposition algorithms, such as cvxEDA [6], utilize quadratic programming and matrix inversion across the entire recording session. Because cvxEDA requires non-causal future samples and consumes seconds of CPU time per window, it cannot be deployed on real-time wearable microcontrollers. A causal, real-time decomposition is strictly necessary.")
p("Furthermore, inter-individual baseline skin conductance varies by over 500% across subjects. A model operating without personalized calibration suffers from acute subject-level failure, while models trained with whole-session normalization leak future test data. A principled solution must utilize an isolated resting calibration baseline.")
h("2.4   Research gap addressed in this work")
p("This project addresses the architectural, signal-processing, and evaluation gaps that arise when deploying multi-signal health monitoring on edge hardware under real-world wearable constraints. We establish four hardware-grounded, causally sound pipelines, eliminate duration shortcuts and normalization leaks, and benchmark the complete multi-task workload on the NVIDIA Jetson Orin Nano.")

# --- PAGE 16 (5): CHAPTER 3 - THEORETICAL STUDY: ECG PIPELINE (Part 1) ---
page("Theoretical Study", "Chapter 3")
h("3.1   ECG voltage and morphology: cardiac strain and LVH")
p("Left Ventricular Hypertrophy (LVH) represents a pathological thickening of the myocardium of the heart's left ventricle, predominantly induced by chronic hypertension and arterial stiffness in aging adults. It serves as a major clinical precursor to congestive heart failure, ventricular arrhythmias, and ischemic stroke. Clinicians identify LVH on 12-lead electrocardiograms through increased QRS complex voltages and repolarization ST-T strain patterns.")
h("The Voltage Preservation Novelty")
p("Conventional deep learning pipelines for ECG classification apply zero-mean unit-variance z-score normalization to every recording. In LVH detection, z-scoring is fatal: it divides the waveform by its standard deviation, erasing the primary clinical biomarker—absolute millivolt R-wave amplitude! Our pipeline maintains calibrated physical units (mV) throughout ingestion and filtering.")
h("Rational Polyphase FIR Resampling (500 Hz to 360 Hz)")
p("PTB-XL records are sampled at 500 Hz, whereas edge wearable hardware operates at 360 Hz. Resampling is implemented via an exact rational factor of L/M = 18/25. Input samples are upsampled by 18, passed through a causal Kaiser-window low-pass filter (beta = 5.0, cutoff fc = 162 Hz, attenuation > 65 dB), and decimated by 25. The exact 10.0-second window duration (3,600 samples) and true millivolt amplitudes are preserved without phase distortion.")

# --- PAGE 17 (6): CHAPTER 3 - THEORETICAL STUDY: ECG PIPELINE (Part 2) ---
page("Theoretical Study (continued)", "Chapter 3")
h("Causal Signal Quality Gate and Morphology Branch")
p("A hardware gate validates sample integrity before neural execution: non-finite (NaN/Inf) samples are rejected, clock jitter is constrained (|dt - 1/fs| <= 0.2/fs), and flatlines (std < 10^-5 mV) or ADC clipping rails are flagged. Corrupted frames emit an 'insufficient_signal' status, preventing erroneous inferences. In parallel, a morphology branch extracts QRS amplitudes and ST elevation/depression proxies.")
gap(4)
fig(thesis_figs / "flow_ptbxl_ecg_lvh.png", "Figure 3.1   End-to-end physiological pipeline for Lead-II ECG LVH detection", max_h=190)
gap(4)
fig(results / "20261105_lvh_comparison/preprocessing/preprocessing_example.png", "Figure 3.2   Calibrated Lead-II ECG waveform preservation and baseline tracking under synthetic noise", max_h=130)

# --- PAGE 18 (7): CHAPTER 3 - THEORETICAL STUDY: PPG PIPELINE (Part 1) ---
page("Theoretical Study", "Chapter 3")
h("3.2   Optical pulse and breathing: respiratory modulation")
p("Obstructive Sleep Apnea (OSA) involves recurrent collapses of the pharyngeal airway during sleep, leading to transient arterial hypoxemia, sympathetic surges, and sleep fragmentation. Polysomnography (PSG) requires extensive wired monitoring in a specialized clinic. Photoplethysmography (PPG) offers an unobtrusive alternative by measuring blood volume changes via optical reflectometry (Red 660 nm and IR 880 nm).")
h("NOVELTY: Pair-Mean Preserving Haar Wavelet Shrinkage")
p("Optical PPG carries high-frequency cardiac pulses (~1-2 Hz) alongside slow respiratory baseline modulation (~0.1-0.3 Hz). Naive high-pass filtering erases the respiratory envelope. Our pipeline applies Haar soft-threshold shrinkage to adjacent optical sample pairs: detail coefficients are attenuated by 0.5 * MAD, while the pairwise sample mean (x1 + x2)/2 is strictly preserved, damping high-frequency sensor noise while protecting respiratory modulations.")
h("Multi-Rate Synchronized View Extraction (31 features/second)")
p("The pipeline extracts 31 synchronized features per second across three distinct views:")
p("• View 1: Pulse Morphology (16 features) — 0.45-5.5 Hz Butterworth bandpass, normalized pulse interpolated into 16 shape coordinates per second scaled by a 45 s trailing pulse amplitude reference.")
p("• View 2: Slower Respiratory Modulation (11 features) — 0.08-0.38 Hz respiratory filtering, baseline drift, pulse amplitude variation (AM), interbeat interval (IBI) variation (FM), and 8 s exponential RMS energies.")
p("• View 3: Signal Quality Metrics (4 features) — High-frequency residual RMS, equal-neighbor sample fraction, systolic peak staleness, and ADC rail dwell fraction.")

# --- PAGE 19 (8): CHAPTER 3 - THEORETICAL STUDY: PPG PIPELINE (Part 2) ---
page("Theoretical Study (continued)", "Chapter 3")
h("Boundary-Preserving Second-by-Second Context")
p("Traditional sleep models assign a single label to an entire 30-second window, blurring onset and recovery boundaries. Our architecture ingests a 30-second context window with an optional 90-second modulation history, but emits five individual second-level predictions at each 5-second step, preserving exact clinical event boundaries.")
gap(6)
fig(thesis_figs / "flow_osasud_ppg_apnea.png", "Figure 3.3   End-to-end physiological pipeline for optical PPG sleep apnea and hypopnea detection", max_h=230)

# --- PAGE 20 (9): CHAPTER 3 - THEORETICAL STUDY: IMU PIPELINE (Part 1) ---
page("Theoretical Study", "Chapter 3")
h("3.3   Impact and body motion: weak supervision and IPMV")
p("Falls represent the leading cause of fatal and non-fatal accidental injuries among older adults. A fall detection system must differentiate between true high-impact falls and vigorous activities of daily living (ADLs), such as sitting down quickly, stumbling, or ascending stairs.")
h("Eliminating the File Duration Leak")
p("As established in our audit, SisFall trials possess an artificial duration cue (~15 s for falls vs up to 180 s for ADLs). To enforce absolute clinical validity, our Impact-Preserving Multi-View (IPMV) pipeline partitions all recordings into fixed, unpadded 2.0-second streaming windows (40 tokens, 50 ms hop), eliminating duration as an information channel.")
h("Native 200 Hz Impact Peak Preservation Path")
p("Impact spikes during a fall typically last only 10 to 30 milliseconds. Standard decimation or fixed averaging filters severely smooth and attenuate peak acceleration. IPMV establishes a dual-path architecture: native-rate 200 Hz samples feed directly into peak derivative (max |da/dt|) and extrema trackers, while an adaptive 30 Hz Butterworth filter increases raw signal pass-through whenever acceleration norms surge.")

# --- PAGE 21 (10): CHAPTER 3 - THEORETICAL STUDY: IMU PIPELINE (Part 2) ---
page("Theoretical Study (continued)", "Chapter 3")
h("Dynamic Gravity Decoupling and Coordinate Invariance")
p("Wearable inertial sensors are subject to tilt and orientation shifts when worn on clothing or a belt. To achieve complete mathematical coordinate invariance, a 0.75 Hz 2nd-order filter estimates the 3D gravity vector g. We subtract gravity to obtain dynamic acceleration a_dyn = a - g, and project acceleration and rotation into vertical (parallel to g) and horizontal (perpendicular to g) components. Replay tests across random 3D rotations resulted in 0 decision changes.")
gap(4)
fig(thesis_figs / "flow_sisfall_imu_fall.png", "Figure 3.4   End-to-end physiological pipeline for waist IMU emergency fall detection vs ADL", max_h=190)
gap(4)
fig(results / "sisfall_v1/figures/gravity_components.png", "Figure 3.5   Dynamic gravity decoupling: acceleration projection into vertical and horizontal planes", max_h=130)

# --- PAGE 22 (11): CHAPTER 3 - THEORETICAL STUDY: EDA PIPELINE (Part 1) ---
page("Theoretical Study", "Chapter 3")
h("3.4   Wrist conductance: tonic and phasic responses")
p("Electrodermal Activity (EDA), also referred to as Galvanic Skin Response (GSR), quantifies autonomic sympathetic nervous arousal through changes in electrical skin conductance caused by palmar and wrist sweat gland innervation. Chronic autonomic arousal exacerbates hypertension, sleep fragmentation, and cognitive decline in older adults.")
h("NOVELTY: MC-SAED Causal Asymmetric Decomposition")
p("Skin conductance consists of two superimposed physiological components: a slow, baseline thermal/metabolic drift (< 0.05 Hz, Tonic / SCL) and sharp, transient sympathetic impulses (Phasic / SCR). Standard convex optimization (cvxEDA) is acausal and requires 8,290 ms per minute of signal. We engineered MC-SAED (Motion-Conditioned Sub-harmonic Asymmetric Exponential Decomposition), which enforces an asymmetric tracking rate:")
c("decay_rate > rise_rate")
p("The tonic baseline relaxes downward quickly as sweat reabsorbs, but follows upward surges very slowly, preventing sympathetic SCR peaks from artificially biasing the baseline. MC-SAED is fully causal, stateful, and executes in 0.26 ms per minute of signal on edge hardware.")

# --- PAGE 23 (12): CHAPTER 3 - THEORETICAL STUDY: EDA PIPELINE (Part 2) ---
page("Theoretical Study (continued)", "Chapter 3")
h("Personalized 120-Second Resting Baseline Calibration")
p("Inter-individual resting skin conductance varies by over 500% across subjects. Rather than applying whole-session normalization (which causes data leakage), our pipeline records a single 120-second resting calibration capture per subject. Features are scaled as personal robust z-scores relative to rest: rel_tonic = (tonic - median(rest)) / (1.4826 * MAD(rest)), improving held-out macro F1 by +0.095 without leaking future session data.")
gap(4)
fig(thesis_figs / "flow_wesad_eda_stress.png", "Figure 3.6   End-to-end physiological pipeline for wrist EDA autonomic stress detection", max_h=190)
gap(4)
fig(results / "wesad_loso_v1/preprocessing/protocol_examples.png", "Figure 3.7   MC-SAED tonic and phasic decomposition across baseline, stress, amusement, and meditation", max_h=130)

# --- PAGE 24 (13): CHAPTER 4 - EXPERIMENTAL STUDY: ECG (Part 1) ---
page("Experimental Study", "Chapter 4")
h("4.1   ECG model comparison on official PTB-XL patient folds")
p("We evaluated single-lead Lead-II models across five random seeds on official PTB-XL patient partitions. Folds 1-6 trained the backbones, folds 7-8 selected epochs, fold 9 calibrated logistic thresholds, and fold 10 served as the frozen test set. Two distinct tasks were examined: LVH vs normal controls (strict) and LVH vs broad clinical rest (reflecting real-world screening).")
gap(4)
table("4.1  Lead II ECG models across two negative-class definitions on Fold 10",
      ["Negatives", "Architecture", "AUROC", "AUPRC", "Sensitivity", "Specificity", "F1"],
      [
          ["normal", "1D Residual CNN", "0.933", "0.884", "82.2%", "90.8%", "0.775"],
          ["normal", "CNN + SSM (State-Space)", "0.900", "0.837", "76.3%", "90.5%", "0.735"],
          ["normal", "CNN + Transformer", "0.910", "0.851", "77.1%", "91.2%", "0.749"],
          ["rest", "1D Residual CNN", "0.811", "0.402", "52.0%", "89.2%", "0.413"],
          ["rest", "CNN + SSM (State-Space)", "0.792", "0.378", "47.5%", "89.5%", "0.388"],
          ["rest", "CNN + Transformer", "0.787", "0.378", "47.7%", "89.5%", "0.389"]
      ],
      [0.12, 0.28, 0.12, 0.12, 0.12, 0.12, 0.12])
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/model_comparison.png", "Figure 4.1   AUROC, AUPRC, and Sensitivity comparisons for Lead-II ECG architectures", max_h=120)
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/roc_pr_calibration.png", "Figure 4.2   ROC, precision-recall, and Platt calibration curves on frozen test fold 10", max_h=120)

# --- PAGE 25 (14): CHAPTER 4 - EXPERIMENTAL STUDY: ECG (Part 2) ---
page("Experimental Study (continued)", "Chapter 4")
h("Privileged Multi-Lead Knowledge Distillation")
p("To bridge the gap between 12-lead clinical diagnostics and single-lead wearable hardware, we trained a privileged 12-lead Inception teacher (E2_inception, achieving 0.734 AP) and distilled its representations into a compact single-lead student (prep_minimal, 9,450 parameters). On the broad clinical rest task, prep_minimal achieved 0.437 patient AP, outperforming the raw uncalibrated baseline while executing in 0.43 ms on edge hardware.")
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/confusion_matrices.png", "Figure 4.3   Ensemble confusion matrices on fold 10 (threshold selected on fold 9)", max_h=130)
gap(4)
fig(results / "lvh_distill_v1/heldout_curves.png", "Figure 4.4   Privileged 12-lead teacher vs single-lead distilled student discrimination curves", max_h=135)

# --- PAGE 26 (15): CHAPTER 4 - EXPERIMENTAL STUDY: PPG (Part 1) ---
page("Experimental Study", "Chapter 4")
h("4.2   Patient-held-out respiratory event detection on OSASUD")
p("OSASUD evaluation utilized a strict 5-fold cross-validation scheme: 18 patients trained the networks, 6 validated thresholds, and 6 untouched patients formed the test set. Out of 961,357 recorded seconds, 821,545 were evaluated across 5,985 gold-standard respiratory events.")
gap(4)
table("4.2  Respiratory event detection models on held-out OSASUD patients",
      ["Model Architecture", "Patient AP", "Event Recall", "Event F1", "False alerts/hour"],
      [
          ["CNN + Transformer", "0.238", "52.2%", "0.252", "68.8 / h"],
          ["TCN + GRU (Multitask)", "0.234", "62.4%", "0.281", "73.8 / h"],
          ["Wavelet + State Space", "0.228", "60.3%", "0.266", "76.9 / h"],
          ["Binary TCN + GRU", "0.242", "57.6%", "0.276", "68.2 / h"],
          ["Dual-Rate TCN+GRU (dual_hard)", "0.247", "46.3%", "0.257", "56.2 / h"],
          ["Dual-Rate Auxiliary Fraction", "0.248", "45.1%", "0.264", "51.6 / h"]
      ],
      [0.34, 0.16, 0.16, 0.16, 0.18])
gap(4)
fig(results / "osasud_v1/figures/models.png", "Figure 4.5   Held-out patient average precision and F1 scores across primary PPG models", max_h=120)
gap(4)
fig(results / "osasud_v1/figures/pr_calibration.png", "Figure 4.6   Precision-recall and calibration curves for optical respiratory event scoring", max_h=120)

# --- PAGE 27 (16): CHAPTER 4 - EXPERIMENTAL STUDY: PPG (Part 2) ---
page("Experimental Study (continued)", "Chapter 4")
h("Dual-Rate Follow-Up and Alarm Burden Analysis")
p("The two-time-scale hard-label model (dual_hard, 9,097 parameters) emerged as the superior research candidate for edge deployment. By decoupling fast pulse shape (12 s) from slow respiratory modulations (90 s), it generated 17.7 fewer false alerts per hour (56.2/h vs 73.8/h for multitask TCN+GRU). At strict low-alarm operating points (<= 1 false alarm/hour), event sensitivity drops to 0.70%, proving that unattended caregiver alarming remains unfeasible without secondary physiological confirmation.")
gap(4)
fig(results / "osasud_binary_dual_rate_v1/models.png", "Figure 4.7   Dual-rate model comparison: Patient AP, Event F1, and False Alarms/hour", max_h=130)
gap(4)
fig(results / "osasud_binary_dual_rate_v1/alert_budgets.png", "Figure 4.8   Operating trade-off curves under validation allowances of 1, 10, and 30 false events/hour", max_h=135)

# --- PAGE 28 (17): CHAPTER 4 - EXPERIMENTAL STUDY: IMU (Part 1) ---
page("Experimental Study", "Chapter 4")
h("4.3   Subject-independent fall trial classification on SisFall")
p("SisFall was evaluated across five outer participant groups containing 38 subjects (23 young adults, 15 older adults) over 4,500 trials (2,702 ADLs, 1,798 falls). Training enforced fixed 2.0-second unpadded streaming windows to eliminate the file duration leak.")
gap(4)
table("4.3  Selected primary SisFall models in the matched 200 Hz, 2.0 s screen",
      ["Model Architecture", "Subject BA", "Fall Recall", "ADL Specificity", "Fall F1", "False Alerts/h"],
      [
          ["Compact InceptionTime", "98.48%", "98.22%", "98.70%", "98.14%", "2.41 / h"],
          ["100 Hz IPMV + XGBoost", "98.51%", "98.72%", "98.11%", "98.04%", "3.59 / h"],
          ["Random Forest (168 feats)", "98.44%", "98.50%", "98.33%", "98.01%", "3.52 / h"],
          ["RBF-SVM (168 feats)", "98.35%", "98.28%", "98.33%", "97.89%", "3.93 / h"],
          ["Dual CNN-TCN + Attention", "92.68%", "86.15%", "99.15%", "91.93%", "1.93 / h"],
          ["Distilled DM-TCN (3 seeds)", "98.61%", "98.61%", "98.61%", "98.88%", "2.21 / h"]
      ],
      [0.30, 0.14, 0.14, 0.14, 0.14, 0.14])
gap(4)
fig(results / "sisfall_v1/figures/model_comparison.png", "Figure 4.9   Held-out subject balanced accuracy across 22 candidate inertial architectures", max_h=120)
gap(4)
fig(thesis_figs / "sisfall_top_three_models.png", "Figure 4.10   Visual comparison of the top three matched SisFall models: InceptionTime, XGBoost, and RF", max_h=120)

# --- PAGE 29 (18): CHAPTER 4 - EXPERIMENTAL STUDY: IMU (Part 2) ---
page("Experimental Study (continued)", "Chapter 4")
h("Distillation Stability and Perturbation Replay")
p("Training DM-TCN with trial-only loss resulted in extreme seed instability (varying from 82.62% to 99.04% BA). Distillation using a weak XGBoost witness teacher (soft targets T=2) stabilized neural convergence across all seeds to 98.61% mean BA with only 2.21-3.10 false alerts/h. In replay tests under simulated sensor perturbations (3D coordinate rotation, gain shifts, bandwidth clipping), the dynamic gravity projection ensured complete stability.")
gap(4)
fig(results / "sisfall_v1/figures/confusion.png", "Figure 4.11   Nested-selection confusion matrices: all candidates (left) vs neural menu (right)", max_h=130)
gap(4)
fig(results / "sisfall_v1/figures/transfer_distilled.png", "Figure 4.12   Distilled DM-TCN replay robustness under sensor rotation, noise, gain, and bandwidth clipping", max_h=135)

# --- PAGE 30 (19): CHAPTER 4 - EXPERIMENTAL STUDY: EDA (Part 1) ---
page("Experimental Study", "Chapter 4")
h("4.4   Wrist sensor constraint and calibration on WESAD")
p("WESAD was evaluated across 15 healthy participants using 15-fold Leave-One-Subject-Out (LOSO) cross-validation across 1,008 baseline and 628 acute TSST stress windows. Within each fold, 12 subjects trained the models, 2 tuned hyperparameters, and 1 untouched subject was tested.")
gap(4)
table("4.4  Participant means on matched WESAD EDA windows (LOSO)",
      ["Model Architecture", "Inputs", "Resting Cal", "Macro F1", "Sensitivity", "Specificity", "AUROC"],
      [
          ["Calibrated Logistic", "EDA+ACC", "No (Zero)", "0.873", "82.9%", "92.4%", "0.966"],
          ["Reliability Fusion", "EDA+ACC", "Yes (Rest)", "0.899", "87.9%", "93.0%", "0.986"],
          ["Physics-Guided MSTN", "EDA+ACC", "Yes (Rest)", "0.887", "84.2%", "95.1%", "0.968"],
          ["1D CNN (Wrist EDA)", "EDA only", "Yes (Rest)", "0.829", "82.3%", "88.2%", "0.962"],
          ["1D CNN (Wrist EDA)", "EDA only", "No (Zero)", "0.735", "74.8%", "79.1%", "0.932"],
          ["RBF-SVM (Full Wrist)", "Full Wrist", "Yes (Rest)", "0.909", "89.0%", "93.5%", "0.991"]
      ],
      [0.26, 0.14, 0.14, 0.12, 0.12, 0.12, 0.10])
gap(4)
fig(results / "wesad_loso_v1/eda_only_architectures.png", "Figure 4.13   Macro F1 scores for wrist EDA-only architectures under zero-calibration protocol", max_h=120)
gap(4)
fig(results / "wesad_loso_v1/architectures.png", "Figure 4.14   Multimodal architecture rankings under combined wrist EDA and accelerometry", max_h=120)

# --- PAGE 31 (20): CHAPTER 4 - EXPERIMENTAL STUDY: EDA (Part 2) ---
page("Experimental Study (continued)", "Chapter 4")
h("Schedule Confound and Component Ablations")
p("A critical finding in WESAD is that a timing-only model (predicting stress solely from elapsed session clock) achieves 0.999 macro F1 due to the standardized laboratory protocol schedule! Our models receive no time input. Component ablations confirmed that personalized resting calibration (+0.095 F1) and motion-dependent quality gating (+0.078 F1) provide the primary defensible improvements.")
gap(4)
fig(results / "wesad_loso_v1/rest_roc_pr_calibration.png", "Figure 4.15   Discrimination, precision-recall, and calibration curves under resting-calibrated protocol", max_h=130)
gap(4)
fig(results / "wesad_loso_v1/component_ablations.png", "Figure 4.16   Component ablations: incremental macro F1 gains from tonic/phasic, rest calibration, and motion", max_h=135)

# --- PAGE 32 (21): CHAPTER 4 - JETSON ORIN NANO DEPLOYMENT (Part 1) ---
page("Experimental Study", "Chapter 4")
h("4.5   Edge deployment benchmarking on NVIDIA Jetson Orin Nano")
p("We deployed all primary models on the NVIDIA Jetson Orin Nano Developer Kit (6-core ARM Cortex-A78AE, Ampere GPU) to measure both pure model inference latency and complete end-to-end DSP pipeline execution time (acquisition conversion, FIR filtering, feature extraction, and scoring) over 100 benchmark iterations.")
gap(4)
table("4.5  NVIDIA Jetson Orin Nano latency benchmarks and real-time budget verification",
      ["Model ID", "Modality", "Model Forward (p50/p95)", "End-to-End (p50/p95)", "Hop Deadline", "Budget Used", "Status"],
      [
          ["ecg_prep_minimal", "Lead II ECG", "0.43 / 0.45 ms", "0.90 / 1.02 ms", "2,000 ms", "0.05%", "PASS"],
          ["ecg_lvh_cnn", "Lead II ECG", "3.60 / 3.86 ms", "27.11 / 29.63 ms", "2,000 ms", "1.48%", "PASS"],
          ["ppg_dual_hard", "Optical PPG", "1.17 / 1.31 ms", "15.91 / 22.42 ms", "5,000 ms", "0.45%", "PASS"],
          ["ppg_cnn_trans", "Optical PPG", "0.50 / 0.65 ms", "13.96 / 20.37 ms", "5,000 ms", "0.41%", "PASS"],
          ["imu_native_100", "6-Axis IMU", "0.53 / 0.59 ms", "3.79 / 4.78 ms", "1,000 ms", "0.48%", "PASS"],
          ["imu_dm_distilled", "6-Axis IMU", "0.73 / 0.84 ms", "4.28 / 7.11 ms", "1,000 ms", "0.71%", "PASS"],
          ["eda_logistic", "Wrist EDA", "0.11 / 0.26 ms", "14.38 / 15.63 ms", "15,000 ms", "0.10%", "PASS"],
          ["eda_neural_pg", "Wrist EDA", "0.85 / 0.90 ms", "30.70 / 32.66 ms", "15,000 ms", "0.22%", "PASS"]
      ],
      [0.17, 0.13, 0.18, 0.18, 0.14, 0.10, 0.10])
gap(4)
fig(jetson_dir / "figure1_latency_vs_deadlines.png", "Figure 4.17   Real-time pipeline latency vs real-time hop deadlines on NVIDIA Jetson Orin Nano (log scale)", max_h=190)

# --- PAGE 33 (22): CHAPTER 4 - JETSON ORIN NANO DEPLOYMENT (Part 2) ---
page("Experimental Study (continued)", "Chapter 4")
h("End-to-End Jetson Performance and Dedicated SpO2 Pipeline")
p("Figure 4.18 illustrates the latency breakdown across preprocessing, model forward passes, and post-processing calibration. Figure 4.19 displays end-to-end multimodal confusion matrices executed on the Jetson Orin Nano runtime. Figure 4.20 evaluates Bhanu Pratap's dedicated SpO2 sleep apnea detector package (CNN + NumPy GBM ensemble), achieving 84.2% sensitivity and 0.871 AUROC.")
gap(3)
fig(jetson_dir / "figure2_ablation_comparison.png", "Figure 4.18   Component-wise execution time breakdown on Jetson Orin Nano across modalities", max_h=110)
gap(3)
fig(jetson_dir / "figure3_all_confusion_matrices.png", "Figure 4.19   Complete multimodal confusion matrices evaluated on NVIDIA Jetson Orin Nano runtime", max_h=110)
gap(3)
fig(jetson_dir / "figure4_bhanu_spo2_comparison.png", "Figure 4.20   Dedicated SpO2 sleep apnea detector evaluation on Jetson (CNN vs GBM vs Ensemble)", max_h=110)

# --- PAGE 34 (23): CHAPTER 5 - CONCLUSIONS (Part 1) ---
page("Conclusions and Future Scope of Study", "Chapter 5")
h("5.1   Comprehensive project synthesis and research milestones")
p("This undergraduate research project developed and benchmarked an edge-deployable multimodal physiological monitoring system tailored for elderly healthcare. The system decouples four independently timed sensing pipelines, enforces causal signal processing, eliminates major literature shortcuts (such as the SisFall file-duration leak), and validates real-time execution feasibility on the NVIDIA Jetson Orin Nano edge compute platform.")
gap(4)
table("5.1  System-level synthesis of module results, deployment readiness, and operational caveats",
      ["Physiological Module", "Selected Top Model", "Benchmark Performance", "Deployment Status", "Principal Operational Caveat"],
      [
          ["ECG (Lead II)", "Multiscale 1D ResCNN\n(Distilled Student)", "AUROC: 0.933 (normal)\nPatient AP: 0.437 (rest)", "Exported ONNX\n(0.90 ms E2E)", "Clinician statement, not anatomical echo; wearable Lead II has lower sensitivity."],
          ["PPG (Optical)", "Dual-Rate TCN+GRU\n(dual_hard)", "Patient AP: 0.247\nEvent Recall: 46.3%", "Exported ONNX\n(15.9 ms E2E)", "Optical pulse is indirect; alarm burden (56.2/h) too high for unattended alerts."],
          ["IMU (6-Axis)", "100 Hz IPMV + XGBoost\n(Distilled DM-TCN)", "Subject BA: 98.51%\nFall Recall: 98.72%", "8 KB NumPy Model\n(3.79 ms E2E)", "Scripted laboratory trials; real-world elderly fall mechanics and alert burden unverified."],
          ["EDA (Conductance)", "Calibrated Logistic\n(Reliable Fusion)", "Macro F1: 0.873 (zero)\nMacro F1: 0.899 (rest)", "3.6 KB JSON Model\n(14.4 ms E2E)", "Laboratory TSST schedule confound; external ambulatory validation outstanding."]
      ],
      [0.17, 0.20, 0.21, 0.18, 0.24])
p("Across all four modalities, mathematical verification confirmed that edge inference consumes less than 2.0% of available hop deadlines, proving that real-time multi-signal edge monitoring is computationally viable on low-power ARM/NVIDIA architectures.")

# --- PAGE 35 (24): CHAPTER 5 - CONCLUSIONS (Part 2) ---
page("Conclusions and Future Scope of Study (continued)", "Chapter 5")
h("5.2   Staged translational roadmap and future priorities")
p("Translating this research prototype into clinical practice requires a disciplined, four-phase engineering roadmap:")
p("1. Hardware Calibration & Sensor Parity: Bench-calibrate the physical BioAmp EXG Pill, MAX30102 optical sensor, waist MPU-9250, and wrist GSR electrodes on physical test rigs to measure exact ADC noise floors, clock jitter, input-referred millivolts, and contact impedances.")
p("2. Prospective Older-Adult Cohort Study: Conduct protocol-approved clinical data collection across elderly individuals with chronic age-related comorbidities (arrhythmias, vascular stiffness, frail gait), acquiring gold-standard matched reference labels.")
p("3. False Alarm Mitigation & Tiered Alerts: In unattended monitoring, high false-alarm rates cause caregiver alarm fatigue. Future software should implement cross-modal confirmation (e.g. validating an optical apnea desaturation with a concurrent heart-rate dip or validating a fall impact with post-impact immobility).")
p("4. Integrated Firmware & Privacy Governance: Package the runtime into a production Linux systemd service with local SQLite ring buffers, Bluetooth Low Energy (BLE) peripheral broadcasts, and strict data minimization ensuring compliance with medical device standards.")
h("5.3   Concluding assessment")
p("In conclusion, this project demonstrates that robust physiological monitoring on edge hardware is achievable when models are designed with causal discipline, physical-unit preservation, and strict subject independence.")

# --- PAGE 36 (25): REFERENCES ---
page("References")
references_list = [
    "[1]  P. Wagner et al., 'PTB-XL, a large publicly available electrocardiography dataset,' Scientific Data, vol. 7, no. 154, 2020. doi: 10.1038/s41597-020-0495-6.",
    "[2]  N. Strodthoff et al., 'Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL,' IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519-1528, 2021. doi: 10.1109/JBHI.2020.3022989.",
    "[3]  OSASUD data descriptor, 'A dataset of stroke unit recordings for the detection of obstructive sleep apnea syndrome,' Scientific Data, 2022. doi: 10.1038/s41597-022-01272-y.",
    "[4]  A. Sucerquia, J. D. Lopez, and J. F. Vargas-Bonilla, 'SisFall: A Fall and Movement Dataset,' Sensors, vol. 17, no. 1, p. 198, 2017. doi: 10.3390/s17010198.",
    "[5]  P. Schmidt et al., 'Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection,' in Proc. ACM ICMI, 2018. doi: 10.1145/3242969.3242985.",
    "[6]  A. Greco et al., 'cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing,' IEEE Transactions on Biomedical Engineering, vol. 63, no. 4, pp. 797-804, 2016. doi: 10.1109/TBME.2015.2474131.",
    "[7]  X. Yu et al., 'Fall Detection Based on a Hybrid CNN-LSTM Model Using Wearable Inertial Sensor Data,' Frontiers in Bioengineering and Biotechnology, vol. 8, p. 63, 2020. doi: 10.3389/fbioe.2020.00063.",
    "[8]  M. A. Syed et al., 'SisFall fall detection using wavelet features and hierarchical classification,' Sensors, vol. 21, no. 19, p. 6653, 2021. doi: 10.3390/s21196653.",
    "[9]  J. Choi, B. Ahmed, and R. Gutierrez-Osuna, 'Development and Evaluation of an Ambulatory Stress Monitor Based on Wearable Sensors,' IEEE Trans. Inf. Technol. Biomed., vol. 16, no. 2, pp. 279-286, 2012.",
    "[10] A. Taconne, V. D. A. Corino, and L. Mainardi, 'An ECG-Based Model for LVH Detection,' PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11655100/",
    "[11] 'MVKT-ECG: Multi-view Knowledge Transfer for ECG Representation Learning,' arXiv:2301.12178, 2023.",
    "[12] 'ST-MEM: Spatio-Temporal Masked Encoder Modeling for ECG,' in Proc. ICLR, 2024. arXiv:2402.09450.",
    "[13] 'Medformer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification,' in Proc. NeurIPS, 2024. arXiv:2405.19363.",
    "[14] 'SELF-CARE: Context-Aware Stress Detection from Wearable Sensors,' arXiv:2303.08215, 2023.",
    "[15] Project ECG Experiments, 'PTB-XL Lead II comparison and LVH distillation reports,' Research results directories 20261105_lvh_comparison and lvh_distill_v1, NIT Warangal, 2026.",
    "[16] Project Optical Experiments, 'OSASUD respiratory-event study and binary two-time-scale follow-up,' Research results directories osasud_v1 and osasud_binary_dual_rate_v1, NIT Warangal, 2026.",
    "[17] Project Inertial Experiments, 'SisFall subject-held-out study, repeated seeds and transfer reports,' Research results directory sisfall_v1, NIT Warangal, 2026.",
    "[18] Project Wrist EDA Experiments, 'WESAD subject-independent comparison, matched-window controls and verification,' Research results directory wesad_loso_v1, NIT Warangal, 2026.",
    "[19] Project Edge Benchmarks, 'NVIDIA Jetson Orin Nano deployment and benchmark suite,' Deploy directory deploy/jetson, NIT Warangal, 2026.",
    "[20] Elderly Healthcare System Repository, 'Architecture, Data Management, and Deployment Specifications,' Docs directory docs/architecture.md, NIT Warangal, reviewed November 2026."
]
for r in references_list:
    ref(r)

# --- PAGE 37 (26): APPENDIX A - SUPPLEMENTARY EXPERIMENTAL FIGURES (Part 1) ---
page("Experimental figures", "Appendix A")
p("A.1   Demographic and clinical subgroup sensitivity analyses")
p("The panels below document model discrimination across age categories (under 65 vs 65+) and biological sex on PTB-XL Fold 10. While overall AUROC remains stable across age, older adults exhibit higher sensitivity due to higher baseline prevalence of hypertensive cardiac remodelling.")
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/age_subgroups.png", "Figure A.1   Lead-II ECG model sensitivity and specificity broken down by age subgroup (<65 vs 65+)", max_h=130)
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/coverage_and_subgroups.png", "Figure A.2   Selective risk coverage curves and biological sex subgroup sensitivity checks", max_h=130)

# --- PAGE 38 (27): APPENDIX A - SUPPLEMENTARY EXPERIMENTAL FIGURES (Part 2) ---
page("Experimental figures (continued)", "Appendix A")
p("A.2   Algorithmic optimization histories and continuous decomposition benchmarks")
p("Figure A.3 illustrates training and development loss convergence across 5 random seeds for the Lead-II ECG architectures. Figure A.4 benchmarks continuous EDA decomposition: MC-SAED matches cvxEDA and Choi peak detection while executing in 0.26 ms/min (a 30x speedup over cvxEDA).")
gap(4)
fig(results / "20261105_lvh_comparison/plots/lvh_vs_normal/loss_curves.png", "Figure A.3   Training (solid) and development (dashed) BCE loss curves across optimization epochs", max_h=130)
gap(4)
fig(results / "wesad_loso_v1/preprocessing/decomposition_comparison.png", "Figure A.4   Continuous EDA decomposition comparison: Residual RMSE, peak match fraction, and compute time", max_h=130)

# --- PAGE 39 (28): APPENDIX B - REPRODUCIBILITY AND VALIDATION RECORD (Part 1) ---
page("Reproducibility and validation record", "Appendix B")
p("B.1   Verification protocols, data hashes, and software checksums")
p("Every model and experiment in this report is paired with an immutable verification record, participant partition audit, and golden replay checksum. Independent verification ensures that any subsequent run can reproduce identical predictions from raw source files.")
gap(6)
table("B.1  Mandatory verification records, independent units, and golden replay parities",
      ["Physiological Module", "Verification Record Path", "Independent Unit", "Replay Parity Tolerance"],
      [
          ["ECG (Lead II)", "research/results/lvh_distill_v1/final.csv", "Patient and ECG", "Max logit error < 1e-6 (59 records)"],
          ["PPG (Optical)", "research/results/osasud_v1/comparison.csv", "Patient and event", "Max prob error < 6e-8 (real traces)"],
          ["IMU (6-Axis)", "research/results/sisfall_v1/replicate_seeds.csv", "Subject and trial", "Max logit error < 2e-6 (256 windows)"],
          ["EDA (Conductance)", "research/results/wesad_loso_v1/all_models.csv", "Participant (LOSO)", "Exact JSON parameter equivalence"]
      ],
      [0.18, 0.44, 0.20, 0.18])
p("To verify the ECG Lead-II student, filter final.csv to prep_minimal and verify the 90% specificity operating threshold. For PPG, confirm matched event intervals divided by gold-standard PSG intervals. For IMU, average the three replicate seed scores in replicate_seeds.csv. For EDA, verify participant macro F1 in all_models.csv.")

# --- PAGE 40 (29): APPENDIX B - REPRODUCIBILITY AND VALIDATION RECORD (Part 2) ---
page("Reproducibility and validation record (continued)", "Appendix B")
h("B.2   Hardware-software interface contracts and deployment checklist")
p("Physical wearable deployment requires that sensor calibration settings, sampling rates, and filter states are version-controlled alongside neural network weights. A change in electrode placement or ADC gain will alter physical units and invalidate frozen decision thresholds.")
p("1. BioAmp Lead-II Contract: Verified RA negative, LL positive, RL reference. Digital counts converted to physical millivolts via hardware gain factor prior to entering the 360 Hz polyphase FIR resampler.")
p("2. MAX30102 Optical Contract: Operating at 80/100 Hz. Causal Haar shrinkage executed per completed second. History buffer maintains up to 90 seconds of modulation features with real-history validity masks.")
p("3. Waist IMU Contract: Tri-axial acceleration calibrated in g (32/8192 g/LSB), tri-axial angular velocity in deg/s (4000/65536 deg/s/LSB). Streaming windows strictly fixed at 2.0 s (40 tokens, 1.0 s hop) without padding.")
p("4. Wrist EDA Contract: Continuous 4 Hz conductance calibrated in microSiemens (0-100 uS range). Initial 120-second resting baseline capture mandatory before scoring.")
p("5. Jetson Systemd Service: Packaged as 'apnea-detector.service' and 'elderly-monitor.service' for automated startup on boot, with thread-safe IPC queues and local SQLite anomaly logging.")
gap(20)
c("— END OF PROJECT REPORT —")

print(f"Total structured pages created: {len(pages)}")

# -------------------------------------------------------------------------
# COMPILATION: REPORTLAB PDF AND PYTHON-DOCX
# -------------------------------------------------------------------------

# Register fonts
for f in ["times", "timesbd", "timesi"]:
    pdfmetrics.registerFont(TTFont(f, f"C:/Windows/Fonts/{f}.ttf"))

# Setup ReportLab PDF
pdf_path = out_dirs[0] / "Wearable_Health_Monitoring_Progress.pdf"
pdf = Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
pdf.setTitle(title_text)
pdf.setAuthor("Vivek Kumar, Bhanu Pratap, Ujwal Kumar")

# Setup Python-Docx
doc = Document()
doc.core_properties.title = title_text
doc.core_properties.author = "Vivek Kumar, Bhanu Pratap, Ujwal Kumar"

sec = doc.sections[0]
sec.page_width = Mm(210)
sec.page_height = Mm(297)
sec.left_margin = Mm(27)
sec.right_margin = Mm(23)
sec.top_margin = Mm(22)
sec.bottom_margin = Mm(20)

for style in doc.styles:
    if style.type in (1, 2):
        style.font.name = "Times New Roman"
        style.font.size = Pt(14 if style.name.startswith("Heading") or style.name == "Title" else 12)
        style.font.color.rgb = RGBColor(0, 0, 0)
        f_elem = style.element.get_or_add_rPr().rFonts
        if f_elem is not None:
            for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                f_elem.attrib.pop(qn("w:" + attr), None)
            for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
                f_elem.set(qn("w:" + attr), "Times New Roman")

normal_fmt = doc.styles["Normal"].paragraph_format
normal_fmt.space_after = Pt(0)
normal_fmt.line_spacing = Pt(15)

for elem in doc.styles.element.xpath(".//w:pBdr"):
    elem.getparent().remove(elem)

W = 453.5
LEFT = 76.5

frontcount = sum(1 for z in pages if z[3])

def roman(i):
    val_map = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    res = ""
    for val, letter in val_map:
        while i >= val:
            res += letter
            i -= val
    return res

def pg_str(i):
    return roman(i) if i <= frontcount else str(i - frontcount)

def line_draw(txt, kind="p", space=None):
    global y_pos
    bold = kind in ("h", "head", "title")
    is_ref = (kind == "ref")
    size = 14 if bold else (9 if is_ref else 12)
    center = kind in ("title", "head", "c", "cap", "ch")
    default_space = 3 if is_ref else (7 if kind in ("p", "h") else 5)
    after = space if space is not None else default_space
    leading = 17 if bold else (11.5 if is_ref else 15)
    face = "timesbd" if bold else "times"
    para = Paragraph(escape(txt), ParagraphStyle("a", fontName=face, fontSize=size, leading=leading, alignment=1 if center else 0, textColor="#000000"))
    _, ht = para.wrap(W, 1000)
    y_pos -= ht
    para.drawOn(pdf, LEFT, y_pos)
    y_pos -= after

    q = doc.add_paragraph(style={"title": "Title", "head": "Heading 1", "h": "Heading 2"}.get(kind, "Normal"))
    fmt = q.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(after)
    fmt.line_spacing = Pt(leading)
    fmt.keep_with_next = False
    fmt.keep_together = False
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    run = q.add_run(txt)
    run.bold = bold
    run.font.size = Pt(size)

def blank_draw(sz):
    global y_pos
    y_pos -= sz
    q = doc.add_paragraph()
    q.paragraph_format.line_spacing = Pt(sz)
    q.add_run(" ").font.size = Pt(1)

def pic_draw(file_path, ww, hh):
    global y_pos
    im = Image.open(file_path)
    iw, ih = im.size
    scale = min(ww / iw, hh / ih)
    w_draw, h_draw = iw * scale, ih * scale
    y_pos -= h_draw
    pdf.drawImage(ImageReader(str(file_path)), LEFT + (W - w_draw) / 2, y_pos, width=w_draw, height=h_draw)
    y_pos -= 4

    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.line_spacing = 1.0
    q.paragraph_format.space_after = Pt(2)
    q.add_run().add_picture(str(file_path), width=Pt(w_draw), height=Pt(h_draw))

def table_draw(cap, headers, data, widths):
    global y_pos
    if cap:
        line_draw("Table " + cap, "cap")
    tab = doc.add_table(rows=0, cols=len(widths))
    tab.autofit = False
    for col, frac in zip(tab.columns, widths):
        col.width = Pt(W * frac)
    for ri, rr in enumerate([headers] + data):
        cells = tab.add_row().cells
        parts = []
        xx = 0
        for ce, tx, frac in zip(cells, rr, widths):
            w_cell = W * frac
            ce.width = Pt(w_cell)
            q = ce.paragraphs[0]
            q.paragraph_format.line_spacing = Pt(15)
            q.paragraph_format.space_after = Pt(3)
            run = q.add_run(tx)
            run.bold = (ri == 0)
            run.font.size = Pt(11)
            par = Paragraph(escape(tx).replace("\n", "<br/>"), ParagraphStyle("t", fontName="timesbd" if ri == 0 else "times", fontSize=10.5, leading=13))
            _, ht = par.wrap(w_cell - 6, 1000)
            parts.append((par, xx, ht))
            xx += w_cell
        if ri == 0 and cap:
            pdf.setLineWidth(0.7)
            pdf.line(LEFT, y_pos, LEFT + W, y_pos)
        for par, xx, ht in parts:
            par.drawOn(pdf, LEFT + xx + 3, y_pos - ht - 3)
        y_pos -= max(x[2] for x in parts) + 6
        if (ri in (0, len(data))) and cap:
            pdf.line(LEFT, y_pos, LEFT + W, y_pos)
    props = tab._tbl.tblPr
    bd = OxmlElement("w:tblBorders")
    for nm in ["top", "bottom", "insideH"]:
        el = OxmlElement("w:" + nm)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "AAAAAA")
        bd.append(el)
    props.append(bd)
    blank_draw(6)

def entry_draw(txt, num_str):
    global y_pos
    max_len = 70
    display_txt = txt if len(txt) <= max_len else txt[:max_len-3] + "..."
    pdf.setFont("times", 11)
    pdf.drawString(LEFT, y_pos - 12, display_txt)
    pdf.drawRightString(LEFT + W, y_pos - 12, num_str)
    y_pos -= 17

    q = doc.add_paragraph()
    fmt = q.paragraph_format
    fmt.tab_stops.add_tab_stop(Pt(W - 3), WD_ALIGN_PARAGRAPH.RIGHT)
    fmt.line_spacing = Pt(17)
    q.add_run(display_txt + "\t" + num_str)

# Pre-compute TOC and figure mappings
fig_entries = []
tab_entries = []
for k_idx, (p_title, p_sec, p_blocks, p_front) in enumerate(pages, 1):
    for b in p_blocks:
        if b[0] == "fig":
            # b = ('fig', path, caption, max_h)
            fig_entries.append((b[2], pg_str(k_idx)))
        elif b[0] == "table" and b[1]:
            # b = ('table', title, headers, rows, widths)
            tab_entries.append((b[1], pg_str(k_idx)))

print(f"Total Figures registered: {len(fig_entries)}")
print(f"Total Tables registered: {len(tab_entries)}")

# Render Pages
for k, (t, section, blocks, front) in enumerate(pages, 1):
    if k > 1:
        from docx.enum.section import WD_SECTION
        if k == 2 or k == frontcount + 1:
            ns = doc.add_section(WD_SECTION.NEW_PAGE)
            ns.footer.is_linked_to_previous = False
            tag = OxmlElement("w:pgNumType")
            tag.set(qn("w:start"), "2" if k == 2 else "1")
            tag.set(qn("w:fmt"), "lowerRoman" if k == 2 else "decimal")
            ns._sectPr.append(tag)
            foot = ns.footer.paragraphs[0]
            foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
            field = OxmlElement("w:fldSimple")
            field.set(qn("w:instr"), "PAGE")
            foot._p.append(field)
        else:
            doc.add_page_break()

    y_pos = 777
    if section:
        line_draw(section, "ch")
    if t:
        line_draw(t, "head", 14)
    if k > 1:
        pdf.setFont("times", 11)
        pdf.drawCentredString(A4[0] / 2, 38, pg_str(k))
    pdf.bookmarkPage("p" + str(k))
    pdf.addOutlineEntry(t or title_text, "p" + str(k))

    for b in blocks:
        op = b[0]
        if op in ("title", "h", "p", "c", "ref"):
            line_draw(b[1], op)
        elif op == "gap":
            blank_draw(b[1])
        elif op == "logo":
            pic_draw(b[1], 96, 110)
        elif op == "table":
            table_draw(*b[1:])
        elif op == "fig":
            # b = ('fig', file_path, caption, max_h)
            pic_draw(b[1], W, b[3])
            line_draw(b[2], "cap", 4)
        elif op == "toc":
            for j_idx, (tt, ss, _, ff) in enumerate(pages, 1):
                if ff and j_idx > 1 and tt not in ("Contents", "List of Figures (continued)"):
                    entry_draw(tt, pg_str(j_idx))
                elif ss in ("Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"):
                    if not any(pages[prev][1] == ss for prev in range(j_idx - 1)):
                        # First page of chapter
                        clean_ch = ss.replace("  ·  Theoretical Study", "").replace("  ·  Experimental Study", "")
                        entry_draw(f"{clean_ch}   {tt}", pg_str(j_idx))
                elif tt in ("References", "Appendix A", "Appendix B") or ss in ("Appendix A", "Appendix B"):
                    if tt == "References":
                        entry_draw("References", pg_str(j_idx))
                    elif ss.startswith("Appendix A") and not any(pages[prev][1].startswith("Appendix A") for prev in range(j_idx - 1)):
                        entry_draw("Appendix A   Experimental figures", pg_str(j_idx))
                    elif ss.startswith("Appendix B") and not any(pages[prev][1].startswith("Appendix B") for prev in range(j_idx - 1)):
                        entry_draw("Appendix B   Reproducibility and validation record", pg_str(j_idx))
        elif op == "figlist":
            part = b[1]
            per_page = 15
            for cap_txt, p_num in fig_entries[part * per_page : (part + 1) * per_page]:
                entry_draw(cap_txt, p_num)
        elif op == "tableslist":
            for tab_txt, p_num in tab_entries:
                entry_draw(tab_txt, p_num)

    print(f"Page {k:2d} ({t[:28]:<28}) final y = {y_pos:6.1f}")
    if y_pos < 36:
        print(f"  --> WARNING: page {k} ({t}) reached low y={y_pos}")
    pdf.showPage()

# Save PDF & DOCX
pdf.save()
doc.save(out_dirs[0] / "Wearable_Health_Monitoring_Progress.docx")

# Mirror copies
for d in out_dirs[1:]:
    import shutil
    shutil.copy(out_dirs[0] / "Wearable_Health_Monitoring_Progress.pdf", d / "Wearable_Health_Monitoring_Progress.pdf")
    shutil.copy(out_dirs[0] / "Wearable_Health_Monitoring_Progress.docx", d / "Wearable_Health_Monitoring_Progress.docx")

print(f"SUCCESS: Report successfully compiled to {len(pages)} pages!")
print(f"PDF saved to: {pdf_path}")
print(f"DOCX saved to: {out_dirs[0] / 'Wearable_Health_Monitoring_Progress.docx'}")
