#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUD_ID=01

mkdir -p runs

echo "[1/3] torch_npu baseline inference"
python baseline_torch_npu_infer.py --output-dir runs/${RUD_ID}/baseline

echo "[2/3] mindspore target inference"
python target_mindspore_infer.py --output-dir runs/${RUD_ID}/target

echo "[3/3] compare zero-diff alignment"
python compare_inference_outputs.py \
  --baseline runs/${RUD_ID}/baseline \
  --target runs/${RUD_ID}/target \
  --output runs/${RUD_ID}/compare_report.json
