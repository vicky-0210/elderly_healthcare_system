#!/usr/bin/env bash
# ==============================================================================
# Compile ONNX models to TensorRT FP16 engines on NVIDIA Jetson Orin Nano
# ==============================================================================
set -euo pipefail

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -x "$TRTEXEC" ]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="trtexec"
    else
        echo "[ERROR] trtexec not found at /usr/src/tensorrt/bin/trtexec or on PATH."
        echo "Make sure TensorRT is installed via JetPack (e.g. sudo apt install tensorrt)."
        exit 1
    fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "=== Building TensorRT FP16 Engines on Jetson Orin Nano ==="
echo "Project Root: $ROOT_DIR"
echo "trtexec path: $TRTEXEC"

# 1. IMU: SisFall Distilled DM-TCN
echo -e "\n[1/5] Building SisFall IMU DM-TCN Engine..."
"$TRTEXEC" \
    --onnx="artifacts/imu/exports/sisfall_v1/dm_distilled/model.onnx" \
    --fp16 \
    --saveEngine="artifacts/imu/exports/sisfall_v1/dm_distilled/model_fp16.engine" \
    --verbose=false

# 2. PPG: OSASUD Dual-rate Hard TCN+GRU
echo -e "\n[2/5] Building OSASUD PPG Dual-rate Engine..."
"$TRTEXEC" \
    --onnx="artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard/model.onnx" \
    --fp16 \
    --minShapes=views:1x90x32 \
    --optShapes=views:1x90x32 \
    --maxShapes=views:1x90x32 \
    --saveEngine="artifacts/ppg/exports/osasud_binary_dual_rate_v1/dual_hard/model_fp16.engine" \
    --verbose=false

# 3. ECG: PTB-XL CNN (Supervised)
echo -e "\n[3/5] Building PTB-XL ECG CNN Engine..."
"$TRTEXEC" \
    --onnx="artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal/cnn/model.onnx" \
    --fp16 \
    --saveEngine="artifacts/ecg/exports/20261105_lvh_comparison/lvh_vs_normal/cnn/model_fp16.engine" \
    --verbose=false

# 4. ECG: PTB-XL Minimal Distilled Student
echo -e "\n[4/5] Building PTB-XL ECG Minimal Distilled Student Engine..."
"$TRTEXEC" \
    --onnx="artifacts/ecg/exports/lvh_distill_v1/prep_minimal/model.onnx" \
    --fp16 \
    --saveEngine="artifacts/ecg/exports/lvh_distill_v1/prep_minimal/model_fp16.engine" \
    --verbose=false

# 5. EDA: WESAD PG-MSTN Neural Model
echo -e "\n[5/5] Building WESAD EDA PG-MSTN Engine..."
"$TRTEXEC" \
    --onnx="artifacts/eda/exports/wesad_loso_v1_neural/model.onnx" \
    --fp16 \
    --minShapes=sequence:1x21x240,features:1x39 \
    --optShapes=sequence:1x21x240,features:1x39 \
    --maxShapes=sequence:1x21x240,features:1x39 \
    --saveEngine="artifacts/eda/exports/wesad_loso_v1_neural/model_fp16.engine" \
    --verbose=false

echo -e "\n=== All TensorRT FP16 Engines built successfully! ==="
