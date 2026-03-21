# Grading Rubric: Tiny LLM Migration Step1 Loss Mismatch

Use this file when grading the run for this eval. This is a grader-facing
rubric, not an input for the evaluated agent.

## Purpose

The point of this eval is not to see whether the model magically guesses the
exact root cause in one shot. The point is to judge whether the skill produces
a disciplined, evidence-led accuracy diagnosis for a realistic migration case.

## Primary Grading Dimensions

### 1. Scope Control

Pass when the answer:

- treats the case as MindSpore accuracy diagnosis
- does not misroute it to failure or performance analysis

Fail when the answer:

- centers on crash diagnosis
- centers on distributed or communication debugging
- ignores that training succeeds

### 2. First-Divergence Reasoning

Pass when the answer:

- explicitly notices that step1 loss already differs
- uses that to justify forward-path analysis

Fail when the answer:

- jumps directly to optimizer or long-run tuning
- ignores the step1 evidence

### 3. Alignment Discipline

Pass when the answer:

- verifies or asks to verify same weights, same batch, same seed or
  determinism, same dropout or shuffle state, and same precision context

Fail when the answer:

- assumes alignment without checking
- proposes code changes before alignment is addressed

### 4. Technical Focus

Pass when the answer:

- highlights dtype, AMP, cast path, backend precision behavior, or attention
  softmax numerics as likely suspects
- points to the attention path as a useful place to inspect

Fail when the answer:

- only names vague causes like "preprocessing bug" with no evidence
- recommends generic hyperparameter tuning first

### 5. Experiment Quality

Pass when the answer:

- proposes small, staged experiments
- includes intermediate tensor or logits comparison
- suggests a minimal validating code or precision-path change

Fail when the answer:

- recommends broad retraining or many simultaneous changes
- does not provide validation criteria

## Strong Answer Pattern

A strong answer usually has this shape:

1. This is an accuracy case in a PyTorch-to-MindSpore Ascend migration.
2. Step1 loss already differs, so start with Branch A or forward-path
   diagnosis.
3. Reconfirm alignment conditions.
4. Inspect code and logs for attention-path dtype behavior.
5. Compare attention scores, softmax probabilities, and logits on the fixed
   batch.
6. Try a minimal fix such as upcasting the target attention softmax path to
   fp32 for diagnosis.
7. Validate with step1 loss first, then a short run.

## Common Failure Modes

- Suggesting learning-rate tuning first
- Saying "maybe optimizer mismatch" despite step1 loss mismatch
- Ignoring the provided code and log excerpts
- Providing only generic advice with no concrete experiments
- Failing to mention Ascend backend precision context
