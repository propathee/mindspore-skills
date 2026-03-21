# Prompt: Run The Full Accuracy-Agent Test Suite

Use this prompt when you want another agent to run the complete
`skill-creator` evaluation loop for `accuracy-agent`, including execution,
grading, timing capture, benchmark aggregation, and review report generation.

## Copy-And-Run Prompt

```text
Use `skill-creator` and run the full evaluation workflow end to end for the
MindSpore `accuracy-agent` skill. Do not stop after drafting prompts or after
running only one phase. Complete the whole loop for the current test suite:

Skill under test:
- Skill path: the current `accuracy-agent` skill directory
- Skill file: `SKILL.md`

Skill-creator resources:
- Skill-creator path: the `skill-creator` skill directory
- Grader instructions: `agents/grader.md`
- Analyzer instructions: `agents/analyzer.md`
- Schema reference: `references/schemas.md`

Eval inputs:
- Evals file: `tests/evals.json`
- Eval metadata template for eval 1:
  `tests/eval-metadata/llm-migration-step1-loss-mismatch.eval_metadata.json`
- Eval 1 hidden judge guide:
  `tests/judge-guides/llm-migration-step1-loss-mismatch.md`
- Eval 1 grading rubric:
  `tests/grading-rubrics/llm-migration-step1-loss-mismatch.md`

Workspace rules:
- Put results in a sibling workspace of the skill directory:
  `../accuracy-agent-workspace`
- Detect the next iteration number automatically. If no prior iteration exists,
  use iteration-1.
- Within the iteration directory, create one directory per eval using a
  descriptive name, not just eval-0.

Run scope:
1. Read the eval set from tests/evals.json.
2. For every eval, spawn both configurations in the same turn:
   - with_skill
   - without_skill
3. Save outputs under:
   - <workspace>/iteration-N/<eval-name>/with_skill/outputs/
   - <workspace>/iteration-N/<eval-name>/without_skill/outputs/
4. Create an eval_metadata.json for every eval directory.
   - For eval 1, start from the provided template and keep the assertions.
   - For any eval that still lacks assertions or has weak ones, draft or refine
     them while runs are in progress, then update both the eval metadata file
     and tests/evals.json if needed.
5. Capture timing.json immediately when each run finishes. Do not wait until the
   end to record timing.
6. After all runs finish, grade every run.
   - Use the grader instructions from skill-creator.
   - For eval 1, use both the hidden judge guide and the grading rubric as
     grader-facing context.
   - For other evals, grade against the expectations in eval_metadata.json and
     the transcript/output evidence.
7. Save grading.json for each run.
8. Aggregate the iteration into benchmark.json and benchmark.md using the
   skill-creator aggregation script.
9. Do an analyzer pass and surface patterns that the raw pass rates might hide.
10. Generate a review report with generate_review.py.
    - If a browser/display is available, launch the normal viewer.
    - If the environment is headless, generate a static HTML review instead.
11. Report back with:
    - where the workspace is
    - where benchmark.json and benchmark.md are
    - where the review report is
    - which evals passed or failed
    - any assertion quality gaps you had to fix
    - whether another iteration is recommended

Execution guidance:
- Follow the `skill-creator` workflow strictly. Do not use /skill-test.
- Keep the whole process in one continuous run until benchmark and review
  artifacts are generated.
- The baseline for this skill evaluation is `without_skill`, not the PyTorch
  model baseline inside the LLM migration case.
- For eval 1, the attached case files are part of the test input. The judged
  agent may inspect them and run them if useful.
- If framework runtimes are missing and the executor cannot actually run the
  training scripts, that is acceptable as long as the executor inspects the
  provided files and log excerpts and the limitation is recorded clearly.
- Use semantic judging for assertions. Do not reduce eval 1 to keyword matching.
- Preserve all generated artifacts needed for later comparison in a future
  iteration.

Required artifacts for each run:
- outputs/
- eval_metadata.json
- timing.json
- grading.json
- any generated metrics.json

Required iteration-level artifacts:
- benchmark.json
- benchmark.md
- analyzer notes or equivalent summary
- review report output

At the end, give a concise summary of results and remaining weak spots in the
test suite itself.
```

## Notes

- This prompt assumes the eval suite lives in
  `tests/evals.json` relative to the `accuracy-agent` skill directory.
- Paths under "Skill-creator resources" are relative to the `skill-creator`
  skill directory.
- Eval 1 already has a richer semantic-judge setup than the other evals.
  The prompt explicitly tells the executing agent to strengthen any weaker eval
  assertions during the run, which matches the `skill-creator` workflow.
- The prompt is written so it can be handed directly to another agent later,
  without needing extra context from this conversation.
