# MiniCode — AI Coding Agent Built From Scratch

> Reference: Claude Code architecture (reverse-engineered from how-claude-code-works),
> implemented in **~4,850 lines of pure Python** with **zero LangChain dependency**.
> 335 tests (1 skipped), all passing.

A ground-up AI Coding Agent with **while-true Agent Loop**, **progressive Skill routing**,
**scoped conflict-aware Memory**, **3-tier Context Compression with measurable benchmarks**,
**persistent task graph**, **concurrent tool execution**, **multi-agent collaboration**,
and **4-layer security**.

---

## 🔥 Key Features

| # | Feature | What it solves |
|---|---------|---------------|
| 1 | **While-true Agent Loop + 4 Recovery Paths** | Model is the sole decision-maker. Recoverable errors (tool failure / prompt-too-long / output-token limit / transient stream error) are retried inside the loop instead of surfacing to the user |
| 2 | **Multi-model Adapter (litellm)** | One codebase supports 100+ providers (Claude/DeepSeek/OpenAI/Gemini). Switch model = change 1 config line |
| 3 | **Skill Progressive Disclosure + 2-stage Routing** | Full skill instructions are stripped from the System Prompt (only a one-line index stays); the model loads one via the `Skill` tool and the result lands at the message tail. Every task also runs recall → rerank, and the top-k candidates are appended to the **end** of the dynamic section — so everything before them stays byte-identical and the prefix cache still hits. Cache hits are read back from the provider (prompt_tokens_details.cached_tokens / Anthropic cache_read_input_tokens), summed per turn and printed — the layout's benefit is measured, not asserted |
| 4 | **Scoped conflict-aware Memory** | Three types (procedural / episodic / profile), SQLite by default, persistent/current-task scopes, BM25 recall, write-time conflict state, read-time labels and verified outcome feedback |
| 5 | **3-tier Context Compression** (+ tool-layer truncation) | Snip (70% usage) → Collapse (85%) → Autocompact (95%). **Snip 86% / Collapse 95% / Autocompact 99.95% compression ratio**. End-to-end Token cost **reduced 63.7%**. Single-result truncation happens in the tool path at >30K chars — a different trigger (one result's absolute size), so it is not counted as a tier |
| 6 | **Multi-Agent (AgentTool)** | explore (read-only) / general (full) / team = **research → coding → testing pipeline**, each step fed the previous step's output. Roles are tool-scoped (research/testing cannot write files). Child agents get isolated contexts and return only summaries; their token/cost is drained back into the parent's budget so `--max-cost` actually covers them. Semaphore(5) cap |
| 7 | **4-layer Security + Plan/Normal Dual Mode** | Rule filter (dangerous commands + path allowlist) → Tool self-check → AI risk classifier → Human confirmation. Every destructive tool (Bash / Write / Edit) reaches L3 — including the ones that discard uncommitted work (`git reset --hard`, `git checkout --`, `git branch -D`); whitelisted read-only Bash commands are the only thing that short-circuits. Every classifier failure path (no model / exception / out-of-vocabulary risk / unparseable JSON) resolves to HIGH, never a silent pass. Plan mode physically removes write tools |
| 8 | **Concurrent Tool Execution** | Tools declaring `is_concurrency_safe` run together via `asyncio.gather`; blocking I/O (file / network / SQLite) is pushed to threads with `asyncio.to_thread`, so reading N files costs 1× latency instead of N× |
| 9 | **Persistent Task Graph** | `TaskCreate / TaskUpdate / TaskList` store dependency-aware tasks in `.minicode/tasks.json`. Dependencies and state transitions are enforced by the store; snapshots survive session restart and context compression |

---

## 📊 Benchmarks (verifiable, run locally)

All numbers from `python benchmarks/record.py` and `python benchmarks/e2e.py`:

> 这些数字可以用 `python benchmarks/verify_claims.py` 一键复核：它重跑两个基准，
> 和 README / DESIGN / 简历里引用的数字逐个对账，**并检查已撤下的旧值没有残留**，
> 不一致就退出码非 0。纯本地确定性计算，不调 API。

```
Token 压缩基准（模拟长会话：读 200 个文件，~58K token）
───────────────────────────────────
原始（无压缩）                 58,030  token
Snip（占位替换）   压缩比 86% →  8,110  token
Collapse（掐头去尾）压缩比 95% → 2,959  token
Autocompact（全量）压缩比 99.95% →   28  token

端到端 Token 成本（模拟读 30 文件，每个 ~2000 token）
───────────────────────────────────
关压缩  累计 input token  1,173,075  →  上下文超 64K 窗口 → 任务失败
开压缩  累计 input token     425,520  →  任务成功
───────────────────────────────────
Token 成本降低：63.7%
```

验证与 Worktree 生命周期基准（脚本化离线场景）：

```shell
python -m benchmarks.verification_worktree_bench --output artifacts
```

```text
场景通过                 4/4
假完成被阻断              1
Worktree 内验收后提交     1
应用前主工作区修改数      0
cherry-pick 冲突回滚      1
自动合并次数              0
```

> **这个数字改过。** 早期版本的 `_snip` 按「结果大小」排序、优先裁掉最大的，
> 省得更狠（74.8%），但代价是把工具结果整批换成占位符——模型随后就没有工作记忆了。
> 现行版本改为**保留最近 5 条完整结果**、只裁更早的（`capabilities/compression.py` 的 `_snip`），
> 单次压缩变弱、触发次数从 6 涨到 25，所以账面从 74.8% 掉到 63.7%。
> **这是有意的取舍：压缩的目的是让模型还能继续干活，不是把 token 省到最低。**
> 两个数字都可用 `python benchmarks/e2e.py` 复现，差别只在 `_snip` 那几行。

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               Application Layer              │
                    │  main.py (CLI / REPL)  │  session_store.py  │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │              Capabilities (5 modules)         │
                    │  ┌───────────┬──────────┬─────────────────┐ │
                    │  │   skill   │  memory  │  multi_agent     │ │
                    │  ├───────────┼──────────┼─────────────────┤ │
                    │  │   security│ compression│                 │ │
                    │  └───────────┴──────────┴─────────────────┘ │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │                Core Layer                     │
                    │  ┌────────────────────┬────────────────────┐ │
                    │  │    agent_loop.py    │    model_adapter   │ │
                    │  │  (while-true + 4    │  (litellm unified) │ │
                    │  │   recovery paths)   │                    │ │
                    │  └──────────┬─────────┴────────────────────┘ │
                    │             │                                │
                    │  ┌──────────▼───────────────────────────────┐│
                    │  │    tools/  files | shell | web | task ... ││
                    │  │         base.py (SkillTool / RecallMem)  ││
                    │  └──────────────────────────────────────────┘│
                    └──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YHLYYE/MiniCode.git
cd MiniCode
pip install -r requirements.txt

# 2. Configure (copy template, fill in your API key)
cp .env.example .env
# Edit .env → set DEEPSEEK_API_KEY=sk-your-key-here

# 3. Run
python main.py                              # Interactive REPL
python main.py "Fix the login bug"          # Single-shot task
python main.py --mode plan "Refactor utils"  # Plan mode (read-only)
python main.py --max-turns 30 --max-cost 10.0

# Optional external verification: completion is blocked until this exits 0
python main.py "Fix the tests" --verify-command "python -m pytest -q"

# Optional: write-capable sub-agents run in isolated Git worktrees
python main.py "Implement the feature" --isolate-subagents

# Optional: run every Bash call in a constrained Docker container
python main.py "Run the tests" --sandbox-bash --sandbox-image alpine:3.20
```

### Available Tools (16 + 3 optional Worktree lifecycle tools)

| Tool | Description |
|------|-------------|
| `Read` / `Write` | File I/O via pluggable `FilesystemBackend` |
| `Edit` | Exact string replacement in a file (no whole-file rewrite) |
| `Bash` | Shell execution with permission review |
| `Grep` / `Glob` | Regex content search / filename-pattern search |
| `WebSearch` / `WebFetch` | Tavily web search / page fetch with HTML→text |
| `TodoWrite` | In-session task list — treated as a long-task anchor: it is excluded from Snip and re-injected after Autocompact, so the plan survives compression |
| `TaskCreate` | Create a persistent task; dependencies must reference existing task IDs |
| `TaskUpdate` | Start or complete a task after dependency and state-machine validation |
| `TaskList` | List tasks, readiness, blockers and status counts; completed tasks are opt-in |
| `Skill` | Progressive skill activation |
| `RecallMemory` | Cross-session experience retrieval |
| `Remember` | Persist knowledge to long-term memory |
| `Agent` | Sub-agent delegation (explore / general / team) |

Optional with `--isolate-subagents`: `WorktreeList`, `WorktreeApply`, `WorktreeRemove`.

`--isolate-subagents` 会为 `general` 与 `team` 的 coding/testing 阶段创建 Git
Worktree；changes are committed to the task branch but are **not auto-merged**.
`explore` stays read-only in the main workspace. Sub-agents still get only the base
toolset and never receive the `Agent` tool, so recursive delegation is impossible.
Read/Write/Edit/Grep/Glob are confined to the Worktree root. Bash starts in the
Worktree and blocks obvious `cd`/`..`/absolute-path escapes, but this is **not an OS
sandbox** unless `--sandbox-bash` is enabled; use the container sandbox when
executing untrusted commands.

When both `--isolate-subagents` and `--verify-command` are set, the same objective
command runs inside each task Worktree before it is committed. Failed verification
returns the test evidence, preserves the Worktree, and does not create a ready-to-apply
commit.

When isolation is enabled, three lifecycle tools are added:

- `WorktreeList` - inspect branch, verification state, commit and changed files;
- `WorktreeApply` - cherry-pick a verified ready branch after user approval;
- `WorktreeRemove` - remove a completed/failed Worktree while retaining its branch.

`WorktreeApply` and `WorktreeRemove` are destructive and pass through MiniCode's
permission confirmation. Unverified branches cannot be applied. The lifecycle index
is stored in `.minicode/worktrees/index.json` under a cross-process lock; corrupt
indexes fail closed. Apply refuses to start while a merge/rebase/cherry-pick is
already in progress.

### Docker Bash sandbox

`--sandbox-bash` replaces the host Bash tool with `DockerBashTool`. Read, Write,
Edit, Grep and Glob remain host-side workspace-scoped tools; only Bash is
containerized by this option.

- the current workspace is bind-mounted at `/workspace`;
- container root filesystem is read-only;
- `/tmp` is a separate bounded `tmpfs`;
- network defaults to `none`;
- CPU, memory and PID limits are enforced;
- capabilities are dropped and `no-new-privileges` is set;
- caller timeouts are capped at 60 seconds;
- timeout, execution error or task cancellation force-removes the named container;
- stdout and stderr are each capped at 1 MB; runaway output is stopped and marked as truncated.

The default image is `alpine:3.20`. For a project requiring Python, pass an image
that contains the required runtime, for example `--sandbox-image python:3.12-slim`.

### Scoped memory

Memory entries carry `scope`, provenance and conflict metadata. Automatic prompt
injection reads only `persistent` entries; raw tool errors, ordinary file-operation
events and task summaries are `current_task` and remain available through an explicit
`RecallMemory` query.

Writes are deterministic: repeated content updates the existing fingerprint; a new
user correction can supersede an older rule; uncertain same-topic entries are retained
and grouped as `conflicted`. Conflicted entries are labeled in the prompt and never
rendered as unconditional instructions. Injected memory gets `success_count` or
`failure_count` according to the final terminal outcome.

This release deliberately does not perform model-based extraction, conditional
reranking or semantic consolidation.

### Persistent task graph

`TaskCreate`, `TaskUpdate` and `TaskList` give the model a durable plan in
`.minicode/tasks.json`. Tasks move only through `pending → in_progress → completed`.
A pending task cannot start until every `blocked_by` task is completed; malformed
files and dangling dependencies fail closed and are never overwritten.

The current open-task snapshot is injected into the dynamic system-prompt section
and is also used as the long-task compression anchor. Completed tasks are excluded
from automatic prompt injection and are available only through
`TaskList(include_completed=True)`.

Plan mode exposes `TaskCreate` and `TaskList`, but not `TaskUpdate`: it can write and
inspect a plan without pretending execution already happened. Main and child agents
share one project-scoped store. The first version is single-process and deliberately
has no claims, leases or background scheduler.

### Execution Modes

- **Normal**: Full tool access
- **Plan**: Read-only — write tools physically removed from schema

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 335 collected: 334 passed, 1 skipped
```

---

## 📁 Project Structure

```
minicode/
├── main.py                     # CLI entry (Click)
├── config.py                   # Model / API / budget config
├── session_store.py            # JSON-based session save/resume
├── .env.example                # Config template (never commit real .env)
├── requirements.txt            # Dependencies: litellm, click, tiktoken
│
├── core/                       # Low-level agent engine
│   ├── agent_loop.py           # While-true loop + state machine + recovery
│   ├── model_adapter.py        # litellm unified Tool Use protocol
│   ├── state.py                # LoopState (immutable, message accumulation)
│   └── tools/                  # 13 tool implementations
│
├── capabilities/               # High-level agent features
│   ├── skill.py                # SkillSystem + progressive disclosure + routing
│   ├── memory.py               # MemoryManager + 3-type MemoryStore
│   ├── multi_agent.py          # AgentTool + team/explore modes
│   ├── security.py             # 4-layer permission review
│   └── compression.py          # 3-tier context compressor + tool-layer truncation
│
├── benchmarks/                 # Reproducible benchmarks
│   ├── record.py               # Compression ratio measurement
│   └── e2e.py                  # End-to-end token cost comparison
│
├── prompt/                     # System prompt builder
│   └── system_prompt.py        # Static + dynamic split, Prefix Cache-friendly
│
├── skills/                     # Skill definitions (markdown)
│   └── code_review.md          # Example skill
│
└── tests/                      # 335 tests
```

---

## 🧠 Design Decisions (what I'd change differently next time)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **No LangChain** | 4,850 lines > 50K framework; every line is traceable | No automatic integration with LangSmith |
| **Lexical BM25 memory recall** | Code fields (paths, errors, function names) are mostly literal; BM25 avoids request-time model cost and nondeterminism | No semantic recall — conditional reranking remains a measured extension |
| **Rule-based Collapse/Autocompact summaries** | Context is already full when compression triggers — can't call LLM | Lower summary quality than LLM-generated |
| **State machine is *inside* while-true** | Model is sole decision-maker, not a programmer-defined FSM | Can't predict which path the agent will take, harder to debug |
| **AgentTool as Tool Use, not subprocess** | No cross-process coordination overhead; child agents get isolated `messages[]` | Parent-child coupling is tighter than MCP would allow |

---

## 📄 License

MIT — do whatever, attribution appreciated.
