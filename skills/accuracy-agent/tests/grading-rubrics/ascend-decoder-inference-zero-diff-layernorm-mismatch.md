# Grading Rubric: Ascend Decoder Inference Zero-Diff LayerNorm Mismatch

Use this file when grading the run for this eval. This is a grader-facing
rubric, not an input for the evaluated agent.

## Purpose

The point of this eval is to judge whether the skill can handle a stricter
than usual migration requirement: exact output equality across a PyTorch +
`torch_npu` baseline and a MindSpore Ascend port for a tiny decoder block.

Inputs and weights are already aligned. The judged answer should not waste the
main diagnosis path on data mismatch guesses.

## Primary Grading Dimensions

### 1. Scope Control

Pass when the answer:

- treats the case as MindSpore accuracy diagnosis
- keeps the focus on inference output mismatch, not crash handling or
  performance work

Fail when the answer:

- routes the case to failure diagnosis
- centers on training instability, optimizer tuning, or distributed issues

### 2. Exact-Alignment Discipline

Pass when the answer:

- explicitly respects the zero-diff requirement
- acknowledges that shared inputs and shared weights are already exactly
  aligned

Fail when the answer:

- says the mismatch is small enough to ignore
- assumes inputs or weights differ without using the supplied evidence

### 3. First-Divergence Reasoning

Pass when the answer:

- frames the mismatch as a forward-path module or output mismatch
- points to the decoder block internals as the next place to inspect

Fail when the answer:

- jumps to optimizer or retraining suggestions
- ignores that this is inference-only

### 4. Technical Focus

Pass when the answer:

- identifies LayerNorm as a likely hotspot
- uses the fact that the MindSpore target uses `mint` operators except for
  `mindspore.nn.LayerNorm`
- mentions Ascend backend precision or operator-semantic differences as useful
  context

Fail when the answer:

- only names vague causes with no code-grounded reasoning
- focuses on unrelated operators before acknowledging the LayerNorm delta

### 5. Experiment Quality

Pass when the answer:

- proposes small validating experiments
- includes intermediate tensor comparison around LayerNorm or nearby outputs
- proposes a minimal fix such as replacing only the LayerNorm implementation

Fail when the answer:

- recommends many unrelated code changes at once
- omits exact-equality validation criteria

## Strong Answer Pattern

A strong answer usually has this shape:

1. This is a MindSpore inference accuracy case on Ascend.
2. Inputs and weights are already exactly aligned, so start in the forward
   path.
3. Because exact equality is required, even tiny drift is a real failure here.
4. Inspect the decoder block implementation and isolate the LayerNorm path
   first.
5. Compare LayerNorm outputs and then the final decoder outputs on the fixed
   batches.
6. Try the smallest fix: replace `mindspore.nn.LayerNorm` with the mint
   version or otherwise isolate just that operator path.
7. Validate by rerunning the same batches and requiring zero output diff.

## Common Failure Modes

- Treating the mismatch as acceptable because it is numerically small
- Ignoring the provided compare script evidence about exact input and weight
  alignment
- Suggesting optimizer, learning-rate, or training-loop changes
- Missing the LayerNorm implementation delta in the target script
- Recommending broad operator rewrites before isolating the single known delta
