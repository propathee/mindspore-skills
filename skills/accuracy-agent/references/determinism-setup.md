# Determinism Setup

Read this file when exact-alignment work depends on deterministic execution.
This matters most for zero-diff or near-zero-diff experiments, where a missing
framework-specific seed or determinism switch can create misleading drift.

The canonical example is
`tests/cases/llm-decoder-inference-zero-diff-case-a/shared_case_assets.py`
in `enable_alignment_determinism()`.

## Common Setup

Apply shared environment and host-side controls first:

- set `PYTHONHASHSEED`
- set `HCCL_DETERMINISTIC=true` when Ascend communication determinism matters
- set `ASCEND_LAUNCH_BLOCKING=1` when synchronous execution helps diagnosis
- seed Python `random`
- seed NumPy

Do not assume these common settings are enough by themselves. The framework in
use still needs its own deterministic setup.

## Torch

When the compared side uses PyTorch, set:

- `torch.manual_seed(seed)`
- `torch.use_deterministic_algorithms(True)`

If the task uses CUDA or other backend-specific deterministic settings beyond
this baseline, mention that uncertainty instead of pretending determinism is
fully guaranteed.

## torch_npu

When the compared side uses `torch_npu`, also set:

- `torch_npu.npu.manual_seed_all(seed)`
- `torch_npu.npu.manual_seed(seed)`

Treat `torch` and `torch_npu` setup as complementary, not interchangeable.
Using only one side of the switches is weaker than the full alignment setup.

## MindSpore

When the compared side uses MindSpore, set:

- `mindspore.set_seed(seed)`
- `mindspore.set_deterministic(True)`

Do not reduce MindSpore determinism to seeding alone. For exact-alignment work,
the deterministic execution switch matters separately.

## How To Use This In Diagnosis

- If the case compares `torch_npu` and MindSpore on Ascend and the user expects
  exact zero-diff, strongly prefer applying the full framework-specific
  determinism setup on both sides before attributing drift to an operator.
- If the case spans heterogeneous hardware or chip families, deterministic
  setup still helps reduce noise, but it does not by itself justify an exact
  zero-diff expectation.
- If determinism setup is partial or unknown, say so explicitly in
  `Alignment Status` and lower confidence in any fine-grained operator blame.
