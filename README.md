# MiniCode — AI Coding Agent Built From Scratch

> Reference: Claude Code architecture (reverse-engineered from how-claude-code-works),
> implemented in **~3000 lines of pure Python** with **zero LangChain dependency**.
> 151 unit tests (1 skipped), all passing.

A ground-up AI Coding Agent with **while-true Agent Loop**, **progressive Skill routing**,
**3-tier Memory system**, **3-tier Context Compression with measurable benchmarks**,
**concurrent tool execution**, **multi-agent collaboration**, and **4-layer security**.

---

## 🔥 Key Features

| # | Feature | What it solves |
|---|---------|---------------|
| 1 | **While-true Agent Loop + 4 Recovery Paths** | Model is the sole decision-maker. Recoverable errors (tool failure / prompt-too-long / output-token limit / transient stream error) are retried inside the loop instead of surfacing to the user |
| 2 | **Multi-model Adapter (litellm)** | One codebase supports 100+ providers (Claude/DeepSeek/OpenAI/Gemini). Switch model = change 1 config line |
| 3 | **Skill Progressive Disclosure + 2-stage Routing** | Full skill instructions are stripped from the System Prompt (only a one-line index stays); the model loads one via the `Skill` tool and the result lands at the message tail. Every task also runs recall → rerank, and the top-k candidates are appended to the **end** of the dynamic section — so everything before them stays byte-identical and the prefix cache still hits |
| 4 | **3-tier Memory (procedural / episodic / profile)** | Pluggable `MemoryStore` backend, pure-Python n-gram vector search (no ChromaDB crash on Windows) |
| 5 | **3-tier Context Compression** (+ tool-layer truncation) | Snip (70% usage) → Collapse (85%) → Autocompact (95%). **Snip 86% / Collapse 95% / Autocompact 99.95% compression ratio**. End-to-end Token cost **reduced 63.7%**. Single-result truncation happens in the tool path at >30K chars — a different trigger (one result's absolute size), so it is not counted as a tier |
| 6 | **Multi-Agent (AgentTool)** | explore (read-only) / general (full) / team = **research → coding → testing pipeline**, each step fed the previous step's output. Roles are tool-scoped (research/testing cannot write files). Child agents get isolated contexts and return only summaries; their token/cost is drained back into the parent's budget so `--max-cost` actually covers them. Semaphore(5) cap |
| 7 | **4-layer Security + Plan/Normal Dual Mode** | Rule filter (dangerous commands + path allowlist) → Tool self-check → AI risk classifier → Human confirmation. Every destructive tool (Bash / Write / Edit) reaches L3 — including the ones that discard uncommitted work (`git reset --hard`, `git checkout --`, `git branch -D`); whitelisted read-only Bash commands are the only thing that short-circuits. Every classifier failure path (no model / exception / out-of-vocabulary risk / unparseable JSON) resolves to HIGH, never a silent pass. Plan mode physically removes write tools |
| 8 | **Concurrent Tool Execution** | Tools declaring `is_concurrency_safe` run together via `asyncio.gather`; blocking I/O (file / network / SQLite) is pushed to threads with `asyncio.to_thread`, so reading N files costs 1× latency instead of N× |

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
```

### Available Tools (13)

| Tool | Description |
|------|-------------|
| `Read` / `Write` | File I/O via pluggable `FilesystemBackend` |
| `Edit` | Exact string replacement in a file (no whole-file rewrite) |
| `Bash` | Shell execution with permission review |
| `Grep` / `Glob` | Regex content search / filename-pattern search |
| `WebSearch` / `WebFetch` | Tavily web search / page fetch with HTML→text |
| `TodoWrite` | In-session task list — treated as a long-task anchor: it is excluded from Snip and re-injected after Autocompact, so the plan survives compression |
| `Skill` | Progressive skill activation |
| `RecallMemory` | Cross-session experience retrieval |
| `Remember` | Persist knowledge to long-term memory |
| `Agent` | Sub-agent delegation (explore / general / team) |

> `Agent` 的 `worktree` 模式已移除（cwd 隔离未实现）；子 Agent 只拿到基础工具集、
> 不含 `Agent` 本身，所以不存在无限递归。

### Execution Modes

- **Normal**: Full tool access
- **Plan**: Read-only — write tools physically removed from schema

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 151 passed, 1 skipped in 1.04s
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
└── tests/                      # 151 unit tests
```

---

## 🧠 Design Decisions (what I'd change differently next time)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **No LangChain** | 3000 lines > 50K framework; every line is traceable | No automatic integration with LangSmith |
| **Pure-Python n-gram vector search** | ChromaDB's ONNX Runtime crashes on Windows access violation; code fields (paths, errors, function names) are mostly literal duplicates anyway | No true semantic search — FAISS at ~500ms/query would be a natural next step |
| **Rule-based Collapse/Autocompact summaries** | Context is already full when compression triggers — can't call LLM | Lower summary quality than LLM-generated |
| **State machine is *inside* while-true** | Model is sole decision-maker, not a programmer-defined FSM | Can't predict which path the agent will take, harder to debug |
| **AgentTool as Tool Use, not subprocess** | No cross-process coordination overhead; child agents get isolated `messages[]` | Parent-child coupling is tighter than MCP would allow |

---

## 📄 License

MIT — do whatever, attribution appreciated.
