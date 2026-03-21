# LLM Judge Guide: Tiny LLM Migration Step1 Loss Mismatch

This file is for semantic grading. Do not provide it as an input file to the
evaluated agent.

## Hidden Ground Truth

The target MindSpore implementation keeps the attention softmax path in compute
dtype, while the known-good PyTorch + `torch_npu` baseline upcasts attention
scores to `float32` before softmax and casts the probabilities back afterward.

This difference is enough to create:

- a step1 loss mismatch
- sharper attention probabilities in the target
- worse short-run convergence on Ascend

The case is intentionally not a trivial API typo or shape bug.

## What A Strong Diagnosis Should Do

The judged answer should cover most of these points:

1. classify the case as an accuracy problem, not a failure or performance issue
2. identify the comparison as PyTorch + Ascend to MindSpore on Ascend
3. explicitly note that step1 loss already differs
4. treat the issue as a forward-path or Branch A style diagnosis first
5. insist on alignment checks before speculation:
   - same weights
   - same first batch
   - same seed or determinism controls
   - same dropout or shuffle behavior
   - same runtime and precision context
6. suspect dtype, AMP, cast path, or backend precision behavior
7. point to the attention path as a likely hotspot
8. suggest comparing attention scores, attention probabilities, or nearby
   intermediate tensors
9. mention Ascend-specific precision behavior as relevant
10. recommend a small validating change instead of broad retuning
11. propose a concrete validation ladder:
    - golden batch
    - step1 loss
    - short run

## Strong Positive Signals

Good answers often say things like:

- "step1 loss mismatch means start with forward-path diagnosis"
- "check attention score and softmax precision path"
- "compare logits or attention tensors before blaming the optimizer"
- "try temporarily upcasting the MindSpore attention softmax path to fp32"
- "validate on the fixed batch before running a longer training loop"

## Weak Or Incorrect Signals

These should count against the answer:

- routing the case to hard-failure diagnosis
- focusing on communication or distributed issues
- recommending learning-rate tuning first
- focusing on optimizer mismatch while ignoring step1 loss
- claiming the root cause is definitively preprocessing with no evidence
- ignoring the supplied code and run excerpts

## Semantic Expectations For Grading

Treat the following as the core semantic assertions:

- The answer recognizes the case as MindSpore accuracy diagnosis.
- The answer recognizes step1 loss mismatch as the first divergence signal.
- The answer routes primarily to forward-path analysis.
- The answer highlights dtype or cast-path investigation.
- The answer inspects or proposes checking the attention softmax path.
- The answer suggests a minimal validating fix or experiment.
- The answer includes concrete validation criteria.
