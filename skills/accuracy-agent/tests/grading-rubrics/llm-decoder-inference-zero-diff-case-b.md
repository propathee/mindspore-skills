# Grading Rubric: LLM Decoder Inference Zero-Diff Case B

Use this file when grading the run for this eval. This is a grader-facing
rubric, not an input for the evaluated agent.

## Purpose

The point of this eval is to judge whether the skill can isolate a single
operator precision issue after the rest of the decoder-block path has already
been aligned. Inputs, weights, and LayerNorm are intentionally not the main
problem in this case.

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
- acknowledges that LayerNorm has already been intentionally aligned

Fail when the answer:

- says the mismatch is small enough to ignore
- assumes inputs, weights, or LayerNorm are still the main issue without using
  the supplied evidence

### 3. First-Divergence Reasoning

Pass when the answer:

- frames the mismatch as a forward-path module or output mismatch
- points to the decoder block internals and especially GELU as the next place
  to inspect

Fail when the answer:

- jumps to optimizer or retraining suggestions
- ignores that this is inference-only

### 4. Technical Focus

Pass when the answer:

- identifies GELU as a likely hotspot
- notices `approximate='none'` as the key likely cause
- avoids blaming LayerNorm after it has already been aligned with
  `mint.nn.LayerNorm`
- mentions Ascend backend precision or operator-semantic differences as useful
  context

Fail when the answer:

- only names vague causes with no code-grounded reasoning
- focuses on LayerNorm or unrelated operators before acknowledging the GELU
  delta

### 5. Experiment Quality

Pass when the answer:

- proposes small validating experiments
- includes intermediate tensor comparison around GELU or nearby outputs
- proposes a minimal fix such as switching to `approximate="tanh"` as the
  recommended option

Fail when the answer:

- recommends many unrelated code changes at once
- omits exact-equality validation criteria

## Strong Answer Pattern

A strong answer usually has this shape:

1. This is a MindSpore inference accuracy case on Ascend.
2. Inputs and weights are already exactly aligned, and LayerNorm is already
   aligned, so start in the forward path.
3. Because exact equality is required, even tiny drift is a real failure here.
4. Inspect the decoder block implementation and isolate the GELU path first.
5. Compare GELU inputs, GELU outputs, and then the final decoder outputs on
   the fixed batches.
6. Try the smallest fix: switch to `approximate="tanh"` as the recommended
   option, or otherwise use an explicitly aligned GELU path.
7. Validate by rerunning the same batches and requiring zero output diff.

## Common Failure Modes

- Treating the mismatch as acceptable because it is numerically small
- Ignoring the provided compare script evidence about exact input and weight
  alignment
- Suggesting optimizer, learning-rate, or training-loop changes
- Missing the `approximate='none'` GELU mismatch in the target script
- Re-blaming LayerNorm even though the case already aligned it
