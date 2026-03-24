# LLM Judge Guide: Ascend Decoder Inference Zero-Diff LayerNorm Mismatch

This file is for semantic grading. Do not provide it as an input file to the
evaluated agent.

## Hidden Ground Truth

The baseline and target scripts intentionally share exactly the same:

- random input batches
- decoder-block weights
- model topology

The intended mismatch comes from the target MindSpore implementation using
`mindspore.nn.LayerNorm` while the rest of the decoder-block path uses
`mindspore.mint` operators. In this eval, exact output equality is the
requirement, so even a small LayerNorm-induced drift is considered a failure.

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
6. inspect the code and notice the LayerNorm implementation delta
7. point to LayerNorm outputs or nearby intermediate tensors as the next
   comparison target
8. mention Ascend operator precision or semantic-path differences as relevant
9. recommend a minimal validating change focused on LayerNorm first
10. validate on the same fixed batches and require zero output mismatch

## Strong Positive Signals

Good answers often say things like:

- "The inputs and weights are already exact, so do not spend the main path on alignment guessing."
- "Because the requirement is zero deviation, the small mismatch still matters."
- "The only obvious implementation delta is `mindspore.nn.LayerNorm`; isolate that first."
- "Capture LayerNorm outputs and the final decoder outputs on the fixed batches."
- "Swap only the target LayerNorm implementation first, then rerun the exact batch set."

## Weak Or Incorrect Signals

These should count against the answer:

- saying the mismatch is acceptable because it is tiny
- focusing on optimizer, gradients, or training stability
- ignoring the compare script evidence about exact shared inputs and weights
- proposing broad operator rewrites before isolating LayerNorm
- missing the Ascend precision-path context

## Semantic Expectations For Grading

Treat the following as the core semantic assertions:

- The answer recognizes the case as MindSpore accuracy diagnosis.
- The answer recognizes exact input and weight alignment.
- The answer treats zero-diff as the acceptance rule.
- The answer routes primarily to decoder forward-path and LayerNorm analysis.
- The answer proposes a minimal LayerNorm-focused experiment or fix.
- The answer includes exact-equality validation criteria.
