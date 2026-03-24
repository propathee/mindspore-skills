# Tiny Decoder Inference Zero-Diff Case B

This case is a small end-to-end accuracy-diagnosis scenario for
`accuracy-agent`.

## Scenario

- Baseline: PyTorch + `torch_npu` on Ascend
- Target: MindSpore + `mint` operators on Ascend
- Task: inference for a tiny transformer decoder block
- Requirement: zero-diff output alignment on the same inputs and same weights
- Status: both sides run successfully, inputs and weights are exactly aligned,
  but the target still shows a non-zero output mismatch

The case simulates a common migration pattern:

- a `torch_npu` inference script is ported to MindSpore
- the port keeps the same decoder block structure and the same shared weights
- much of the main path has already been aligned, but at least one remaining
  forward-path operator or parameter path still introduces drift
- the intended task is to locate the first real mismatch from evidence, rather
  than guess from the case description

## Files

- `shared_case_assets.py`
  - shared config, deterministic batch generation, shared weights, and
    snapshot helpers
- `baseline_torch_npu_infer.py`
  - known-good PyTorch + `torch_npu` decoder-block inference script
- `target_mindspore_infer.py`
  - MindSpore decoder-block inference script for the migrated decoder block
- `compare_inference_outputs.py`
  - verifies exact input and weight alignment, then reports output tensor
    mismatch statistics batch by batch
- `run_scenarios.sh`
  - runs baseline inference, target inference, and comparison in sequence

## What The Evaluated Agent Should Do

The evaluated agent should:

1. recognize this as a MindSpore accuracy diagnosis case
2. treat it as a cross-framework inference mismatch, not a hard failure
3. verify that inputs and weights are already exactly aligned
4. respect the zero-diff acceptance requirement instead of dismissing the
   mismatch as acceptable tolerance
5. identify the decoder block forward path as the earliest useful place to
   inspect, then narrow further from code and tensor evidence
6. recommend the smallest validating experiment first
7. propose a minimal fix only after isolating the first operator or parameter
   mismatch

## Suggested Commands

Baseline:

```bash
python baseline_torch_npu_infer.py --output-dir runs/baseline
```

Target:

```bash
python target_mindspore_infer.py --output-dir runs/target
```

Compare:

```bash
python compare_inference_outputs.py \
  --baseline runs/baseline \
  --target runs/target \
  --output runs/compare_report.json
```

All-in-one:

```bash
bash run_scenarios.sh
```
