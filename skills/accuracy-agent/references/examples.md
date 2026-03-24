# Accuracy-Agent Examples

Read this file only when you want concrete diagnosis patterns or need a worked
scenario to imitate. The main workflow remains in `SKILL.md`.

## Example 1: Step1 Loss Mismatch Against a Trusted Baseline

**User says:**

> With the same weights and the same batch, PyTorch on Ascend gives step1 loss
> `2.1431`, but MindSpore on Ascend gives `2.3128`. Both runs are single-card,
> dropout is disabled, and batch size is 1. I want to know where they first
> diverge.

**Expected behavior:**

- classify this as `step1 loss mismatch`
- confirm weights, input order, precision, and determinism before deeper claims
- treat it as a forward-path problem first
- recommend tensor comparison from coarse modules down to the first mismatch
- avoid optimizer-focused advice

## Example 2: Later Training Divergence

**User says:**

> Step1 loss is aligned with the previous good run, but after around step 50 the
> local norm and loss curve start drifting. Final validation accuracy is much
> worse.

**Expected behavior:**

- classify this as `step1 loss matches but later diverges`
- inspect gradients, local norm, one-step update, optimizer settings, and
  parallel differences
- suggest an `lr=0` or no-update experiment before broad tuning
- avoid re-running forward-only comparisons as the primary path

## Example 3: Crash Misrouted as Accuracy

**User says:**

> Training stops at step 3 with a RuntimeError. After that the log shows NaN in
> the loss.

**Expected behavior:**

- say this is not an accuracy-only entry point because execution failed
- do not enter the non-fatal NaN or Inf branch
- ask for failure evidence or redirect the user to failure diagnosis

## Example 4: Zero-Diff Inputs but Operator Output Drift

**User says:**

> I already compared the decoder block internals. The GELU inputs are identical
> between torch_npu and MindSpore on the same Ascend batch, but GELU outputs
> start drifting from that point. What should I do next?

**Expected behavior:**

- do not stop at "GELU is suspicious"
- inspect whether the two GELU calls use the same API parameters and defaults
- if parameters differ, align them and rerun the original comparison first
- if parameters are aligned, build a standalone GELU repro with multiple input
  shapes, dtypes, and random values
- if standalone random trials do not reproduce the drift, recommend dumping the
  real GELU inputs and outputs from the model and replaying them
- report whether the issue appears deterministic, intermittent, or still
  unproven

## Example 5: Cross-Hardware Mismatch With No Zero-Diff Guarantee

**User says:**

> The same model runs on GPU and Ascend 910B, and the final logits differ more
> than expected. Should I start dumping every operator and try to align them to
> zero?

**Expected behavior:**

- do not assume zero-diff is the right target for GPU versus Ascend
- ask whether the real goal is exact equality or staying within an acceptable
  accuracy budget
- check whether determinism is enabled and whether the user actually has a
  reason to expect identical low-level execution paths
- prioritize finding the earliest materially significant mismatch rather than
  forcing operator-by-operator zero-diff alignment
- reserve heavy single-operator exact-equality experiments for cases where the
  user and platform context justify that target
