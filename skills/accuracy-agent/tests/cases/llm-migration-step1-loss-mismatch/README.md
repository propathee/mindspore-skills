# Tiny LLM Migration Accuracy Case

This case is a small, realistic accuracy-diagnosis scenario for
`accuracy-agent`.

## Scenario

- Baseline: PyTorch + `torch_npu` on Ascend
- Target: MindSpore on Ascend
- Task: next-token training for a tiny decoder-only language model
- Status: both sides train successfully, but the target shows a step1 loss
  mismatch and later short-run degradation

The case is intentionally designed to be:

- real enough to resemble an actual migration debugging task
- simple enough for an early-version diagnostic skill
- subtle enough that the issue is not a one-line API typo

## Files

- `shared_case_assets.py`
  - shared config, fixed batch, deterministic weight generation, and logging
- `baseline_torch_npu_train.py`
  - known-good baseline training script
- `target_mindspore_train.py`
  - MindSpore target training script with an intentional but non-trivial
    numerical accuracy issue
- `baseline_run.log`
  - representative short-run baseline log excerpt
- `baseline_run.log`
  - representative short-run target log excerpt

## What The Evaluated Agent Should Do

The evaluated agent should:

1. recognize this as a MindSpore accuracy case
2. notice that step1 loss already differs
3. treat it as a forward-path diagnosis first
4. inspect code and evidence before guessing
5. recommend small, staged experiments rather than broad tuning

## Suggested Commands

The scripts are designed to be inspectable first and runnable when the
environment supports the required framework.

PyTorch baseline:

```bash
python baseline_torch_npu_train.py --output-dir runs/baseline
```

MindSpore target:

```bash
python target_mindspore_train.py --output-dir runs/target
```

Both scripts print:

- configuration
- environment context
- model parameter summary
- training phase transitions
- per-step loss
- selected attention-path debug statistics

Both scripts also write a `run_summary.json` into the chosen output directory.
