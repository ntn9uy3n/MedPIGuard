#!/bin/bash
# MedPIDefense — Run Full Deep Learning Baseline
# Usage: bash run_baseline.sh [--model all|mlp|lstm_mlp|bilstm|autoencoder]

set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

MODEL="${1:--model} ${2:-all}"

echo "============================================="
echo "MedPIDefense — Deep Learning Baseline Runner"
echo "============================================="
echo "Command: python3 main.py $MODEL"
echo ""

python3 main.py $MODEL

echo ""
echo "Done. Results saved to models_output/baseline_results.json"
