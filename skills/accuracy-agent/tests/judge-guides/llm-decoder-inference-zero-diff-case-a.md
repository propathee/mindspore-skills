# LLM Judge Guide: LLM Decoder Inference Zero-Diff Case A

This file is for semantic grading. Do not provide it as an input file to the
evaluated agent.

## Hidden Ground Truth

The baseline and target scripts intentionally share exactly the same:

- random input batches
- decoder-block weights
- model topology

The intended mismatch comes from leaving `LayerNorm` at framework defaults:

- PyTorch `nn.LayerNorm` defaults `eps=1e-5`
- MindSpore `nn.LayerNorm` defaults `epsilon=1e-7`

The scripts already align GELU to `tanh`, so GELU is not the intended source
of drift in this eval. Exact output equality is the requirement, so even a
small epsilon-induced LayerNorm drift is considered a failure.

## What A Strong Diagnosis Should Do

The judged answer should cover most of these points:

1. classify the case as an accuracy problem, not a hard failure
2. identify the comparison as PyTorch + `torch_npu` on Ascend versus
   MindSpore on Ascend
3. explicitly note that inputs and weights are already exactly aligned
4. respect the exact-equality requirement instead of reframing it as a normal
   tolerance case
5. identify the first useful divergence stage as the decoder forward path or
   final output tensors
6. inspect the code and notice the LayerNorm default epsilon delta
7. point to LayerNorm outputs or nearby intermediate tensors as the next
   comparison target
8. mention Ascend operator precision or semantic-path differences as relevant
9. recommend a minimal validating change focused on LayerNorm first
10. validate on the same fixed batches and require zero output mismatch

## Strong Positive Signals

Good answers often say things like:

- "The inputs and weights are already exact, so do not spend the main path on alignment guessing."
- "Because the requirement is zero deviation, the small mismatch still matters."
- "The likely problem is the default LayerNorm epsilon mismatch, not GELU."
- "Capture LayerNorm outputs and the final decoder outputs on the fixed batches."
- "Swap to `mint.nn.LayerNorm` as the recommended fix, or set `mindspore.nn.LayerNorm(epsilon=1e-5)`, then rerun the exact batch set."

## Weak Or Incorrect Signals

These should count against the answer:

- saying the mismatch is acceptable because it is tiny
- focusing on optimizer, gradients, or training stability
- ignoring the compare script evidence about exact shared inputs and weights
- proposing broad operator rewrites before isolating LayerNorm
- blaming GELU after the scripts already aligned it to `tanh`
- missing the Ascend precision-path context

## Semantic Expectations For Grading

Treat the following as the core semantic assertions:

- The answer recognizes the case as MindSpore accuracy diagnosis.
- The answer recognizes exact input and weight alignment.
- The answer treats zero-diff as the acceptance rule.
- The answer routes primarily to decoder forward-path and LayerNorm epsilon analysis.
- The answer proposes a minimal LayerNorm-focused experiment or fix, preferably `mint.nn.LayerNorm` as the recommended option, or explicit `epsilon` alignment.
- The answer includes exact-equality validation criteria.
