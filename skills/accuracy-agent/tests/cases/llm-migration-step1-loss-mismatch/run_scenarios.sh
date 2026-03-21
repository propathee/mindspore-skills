#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p runs

echo "[30] fp32 reference"
python baseline_torch_npu_train.py \
  --compute-dtype float32 \
  --softmax-mode fp32 \
  --output-dir runs/30_baseline_fp32_ref

python target_mindspore_train.py \
  --compute-dtype float32 \
  --softmax-mode fp32 \
  --optimizer-impl mint \
  --output-dir runs/30_target_fp32_ref

echo "[31] fp32 reference alignment"
python baseline_torch_npu_train.py \
  --compute-dtype float32 \
  --softmax-mode fp32 \
  --alignment-mode \
  --output-dir runs/31_baseline_fp32_ref_align

python target_mindspore_train.py \
  --compute-dtype float32 \
  --softmax-mode fp32 \
  --optimizer-impl mint \
  --alignment-mode \
  --output-dir runs/31_target_fp32_ref_align

echo "[32] default bf16 reference"
python baseline_torch_npu_train.py \
  --compute-dtype bfloat16 \
  --output-dir runs/32_baseline_bf16_default

python target_mindspore_train.py \
  --compute-dtype bfloat16 \
  --optimizer-impl mint \
  --output-dir runs/32_target_bf16_default

echo "[33] bf16 softmax precision difference"
python baseline_torch_npu_train.py \
  --compute-dtype bfloat16 \
  --output-dir runs/33_baseline_bf16_softmax_ctrl

python target_mindspore_train.py \
  --compute-dtype bfloat16 \
  --softmax-mode compute_dtype \
  --optimizer-impl mint \
  --output-dir runs/33_target_bf16_softmax_ctrl

echo "[34] bf16 layernorm eps difference"
python baseline_torch_npu_train.py \
  --compute-dtype bfloat16 \
  --output-dir runs/34_baseline_bf16_ln_eps

python target_mindspore_train.py \
  --compute-dtype bfloat16 \
  --layer-norm-eps 1e-4 \
  --optimizer-impl mint \
  --output-dir runs/34_target_bf16_ln_eps

echo "[35] bf16 AdamW vs nn.Adam optimizer mismatch"
python baseline_torch_npu_train.py \
  --compute-dtype bfloat16 \
  --output-dir runs/35_baseline_bf16_optimizer_impl

python target_mindspore_train.py \
  --compute-dtype bfloat16 \
  --optimizer-impl nn \
  --output-dir runs/35_target_bf16_optimizer_impl

echo "Scenario runs complete."
