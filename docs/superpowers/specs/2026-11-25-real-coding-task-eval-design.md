# Real Coding Task Evaluation Design

## Goal

Build a reproducible end-to-end evaluation for MiniCode using real model calls on fixed coding tasks with hidden acceptance tests.

The result must distinguish implementation test coverage from actual agent task completion. No metric may be filled manually after a run.

## Scope

Included:

- A versioned suite of 20 small Python repositories with buggy or incomplete code.
- One fixed user prompt and one hidden acceptance test per task.
- Temporary Git repositories so each run starts clean and produces a reviewable patch.
- Optional repeated runs for the same task.
- Objective acceptance through pytest after the agent stops.
- JSON and Markdown reports containing raw per-run evidence and aggregate metrics.
- Fixture validation: every task must fail before the reference solution and pass after it.

Excluded:

- LLM judges for dimensions already covered by tests.
- Synthetic user-feedback scores.
- Claims about production deployment or real user traffic.
- Comparing against historical versions without separately archived baseline runs.

## Architecture

`benchmarks/coding_task_set.py` owns immutable task definitions. Each task contains:

- stable task ID;
- category and prompt;
- initial repository files;
- hidden acceptance test;
- reference solution used only to validate the fixture;
- timeout, turn and cost limits.

`benchmarks/real_coding_eval.py` owns execution:

1. Materialize the initial files into a fresh temporary directory.
2. Initialize Git and commit the baseline.
3. Enter the task directory and build MiniCode tools against that workspace.
4. Attach evaluation hooks to collect model/tool lifecycle events.
5. Run the real model until completion, budget or timeout.
6. Copy only the hidden test into the task after the agent has stopped.
7. Run the hidden test and capture exit code and output.
8. Capture `git diff`, turns, tokens, cost, duration, tool count, tool errors and claim/failure status.
9. Write raw JSON and a readable Markdown summary into `artifacts/`.

## Metrics

Per run:

- task ID and repeat index;
- model, start time and duration;
- termination reason;
- hidden-test exit code and output;
- `passed`;
- `claimed_completion`;
- `false_completion = claimed_completion and not passed`;
- turns, input/output tokens, cost;
- model calls, tool calls and tool errors;
- changed files and patch.

Aggregate:

- attempted and passed runs;
- pass rate with raw numerator/denominator;
- false-completion count and rate;
- mean/median turns, tokens, cost and duration;
- results grouped by task category;
- results grouped by task ID so unstable tasks are visible.

## Safety and isolation

- The agent never sees hidden tests because they are copied only after its session ends.
- Each repeat uses a new temporary workspace.
- No result file is written unless the model actually ran; missing credentials are a hard error.
- Test commands use the current Python interpreter and a fixed timeout.
- The evaluator never evaluates the MiniCode repository itself, so agent edits cannot alter hidden tests.

## Reproducibility

- Task definitions are source-controlled and immutable.
- The model identifier, date, repeat count and all limits are recorded in the report.
- Git patch and hidden-test output are stored for every run.
- A separate command validates that all task fixtures are red on baseline and green on the reference solution.

## Acceptance criteria

- `python -m benchmarks.real_coding_eval --validate-fixtures` proves 20/20 fixtures are valid offline.
- `python -m benchmarks.real_coding_eval --limit 1` performs a real model run when credentials exist.
- A missing API key fails clearly and creates no fake metrics.
- Unit tests cover task materialization, hidden-test timing, aggregation and false-completion detection.
- Full project tests remain green.
