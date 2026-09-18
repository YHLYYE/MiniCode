# MiniCode — AI Coding Agent Built From Scratch

> Reference: Claude Code architecture (reverse-engineered from how-claude-code-works),
> implemented in **~2500 lines of pure Python** with **zero LangChain dependency**.
> 47 unit tests, all passing.

A ground-up AI Coding Agent with **while-true Agent Loop**, **progressive Skill routing**,
**3-tier Memory system**, **4-tier Context Compression with measurable benchmarks**,
**multi-agent collaboration**, and **4-layer security**.

---

## 🔥 Key Features

| # | Feature | What it solves |
|---|---------|---------------|
| 1 | **While-true Agent Loop + 4 Recovery Paths** | Model is the sole decision-maker; recoverable errors don't surface to user. Failure rate from 8% → <1% |
| 2 | **Multi-model Adapter (litellm)** | One codebase supports 100+ providers (Claude/DeepSeek/OpenAI/Gemini). Switch model = change 1 config line |
| 3 | **Skill Progressive Disclosure + 2-stage Routing** | System Prompt stays constant → Prefix Cache structure naturally stable. Recall → Rank for >10 skills |
| 4 | **3-tier Memory (procedural / episodic / profile)** | Pluggable `MemoryStore` backend, pure-Python n-gram vector search (no ChromaDB crash on Windows) |
| 5 | **4-tier Context Compression** | Truncation → Snip → Collapse → Autocompact. **Snip 86% / Collapse 95% / Autocompact 99.95% compression ratio**. End-to-end Token cost **reduced 74.8%** |
| 6 | **Multi-Agent (AgentTool)** | explore / general / team modes. Child agents get isolated contexts, only return summaries. Semaphore(5) concurrency cap |
| 7 | **4-layer Security + Plan/Normal Dual Mode** | Rule filter → Tool self-check → AI risk classifier → Human confirmation. Plan mode physically removes write tools |

---

## 📊 Benchmarks (verifiable, run locally)

All numbers from `python benchmarks/record.py` and `python benchmarks/e2e.py`:

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
开压缩  累计 input token     296,058  →  任务成功
───────────────────────────────────
Token 成本降低：74.8%
```

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

### Available Tools (7)

| Tool | Description |
|------|-------------|
| `Read` / `Write` | File I/O via pluggable `FilesystemBackend` |
| `Bash` | Shell execution with permission review |
| `Skill` | Progressive skill activation |
| `RecallMemory` | Cross-session experience retrieval |
| `Remember` | Persist knowledge to long-term memory |
| `Agent` | Sub-agent delegation (explore / general / team) |

> `WebFetch` / `WebSearch` / `TodoWrite` 定义于 `core/tools/`，但未接入
> `main.py` 的工具集；`Agent` 的 `worktree` 模式已移除（cwd 隔离未实现）。

### Execution Modes

- **Normal**: Full tool access
- **Plan**: Read-only — write tools physically removed from schema

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 47 passed in 1.35s
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
│   └── tools/                  # 7 tool implementations
│
├── capabilities/               # High-level agent features
│   ├── skill.py                # SkillSystem + progressive disclosure + routing
│   ├── memory.py               # MemoryManager + 3-type MemoryStore
│   ├── multi_agent.py          # AgentTool + team/explore modes
│   ├── security.py             # 4-layer permission review
│   └── compression.py          # 4-tier context compressor
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
└── tests/                      # 47 unit tests
```

---

## 🧠 Design Decisions (what I'd change differently next time)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **No LangChain** | 2500 lines > 50K framework; every line is traceable | No automatic integration with LangSmith |
| **Pure-Python n-gram vector search** | ChromaDB's ONNX Runtime crashes on Windows access violation; code fields (paths, errors, function names) are mostly literal duplicates anyway | No true semantic search — FAISS at ~500ms/query would be a natural next step |
| **Rule-based Collapse/Autocompact summaries** | Context is already full when compression triggers — can't call LLM | Lower summary quality than LLM-generated |
| **State machine is *inside* while-true** | Model is sole decision-maker, not a programmer-defined FSM | Can't predict which path the agent will take, harder to debug |
| **AgentTool as Tool Use, not subprocess** | No cross-process coordination overhead; child agents get isolated `messages[]` | Parent-child coupling is tighter than MCP would allow |

---

## 📄 License

MIT — do whatever, attribution appreciated.
