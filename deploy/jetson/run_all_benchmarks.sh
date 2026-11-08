#!/usr/bin/env bash
# ==============================================================================
# Run Jetson Orin Nano benchmark suite for all top models
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Ensure Python packages are reachable
export PYTHONPATH="$ROOT_DIR/packages/common/src:$ROOT_DIR/modules/ecg/src:$ROOT_DIR/modules/ecg/experiments/lvh_distillation:$ROOT_DIR/modules/ppg/src:$ROOT_DIR/modules/imu/src:$ROOT_DIR/modules/eda/src${PYTHONPATH:+:$PYTHONPATH}"

ITERATIONS="${1:-100}"
OUTPUT_DIR="$ROOT_DIR/artifacts/benchmarks"
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo " Running Multi-Signal Model Benchmarks on Jetson Orin Nano"
echo " Iterations: $ITERATIONS"
echo "=========================================================="

# 1. Run CPU Benchmark (ARM Cortex-A78AE)
echo -e "\n>>> [1/2] Benchmarking CPU Execution Provider..."
python3 deploy/jetson/benchmark_jetson.py \
    --iterations "$ITERATIONS" \
    --device cpu \
    --output "$OUTPUT_DIR/jetson_cpu_benchmark.json"

# 2. Run CUDA / GPU Benchmark (NVIDIA Ampere GPU) if onnxruntime-gpu is available
if python3 -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" >/dev/null 2>&1; then
    echo -e "\n>>> [2/2] Benchmarking CUDA Execution Provider..."
    python3 deploy/jetson/benchmark_jetson.py \
        --iterations "$ITERATIONS" \
        --device cuda \
        --output "$OUTPUT_DIR/jetson_cuda_benchmark.json"
else
    echo -e "\n>>> [2/2] Note: CUDAExecutionProvider not found in current onnxruntime."
    echo "    To enable GPU acceleration, install onnxruntime-gpu built for JetPack/aarch64."
fi

echo -e "\n=== Benchmark Complete. Results stored in: $OUTPUT_DIR ==="
