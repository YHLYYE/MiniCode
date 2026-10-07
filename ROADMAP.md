# Roadmap

MiniCode is a **personal, single-author** implementation of a Claude Code–style
coding agent. This file exists so that anyone reading the repo can tell, in one
screen, **what is done, what is deliberately not done, and why**.

The order below is "what I would do next and what it would cost", not a wish list.

---

## Shipped (v1)

| Area | What it actually does | Where |
|---|---|---|
| Agent Loop | `while-true` loop, 4 recovery paths (tool error / prompt too long / output truncated / network transient), each with its own state marker | `core/agent_loop.py` |
| Context | 3-tier compression: Snip → Collapse → Autocompact, plus per-tool-result truncation. End-to-end input tokens −63.7% in the benchmark | `capabilities/compression.py` |
| Tools | 12 built-ins, grouped by `check_concurrency_safe()`; blocking I/O pushed to threads | `core/tools/` |
| Multi-Agent | `AgentTool` routing: `explore` / `general` / `team` (research → code → verify pipeline) | `capabilities/multi_agent.py` |
| Skills | Progressive disclosure (full text stays out of the system prompt) + two-stage routing; top-k appended last so the prefix stays byte-identical | `capabilities/skill.py` |
| Memory | Three types (procedural / episodic / profile), SQLite by default, pluggable store | `capabilities/memory.py` |
| Security | 4 layers; every classifier failure path resolves to HIGH; Plan/Normal dual mode physically removes write tools | `capabilities/security.py` |
| Cost | Per-model pricing via litellm, `--max-cost` circuit breaker, sub-agent usage rolled up into the parent ledger | `core/model_adapter.py`, `core/state.py` |
| Quality gates | 252 tests, pyflakes gate, and `benchmarks/verify_claims.py` re-checking 14 documented claims against the code | `tests/`, `benchmarks/` |

---

## Known boundaries (deliberately not done)

These are **stated on purpose**. A boundary that is written down is engineering;
one that is hidden is a demo.

| Boundary | Why it is not done | What closing it would cost |
|---|---|---|
| **No MCP support** | Tools are built in. MCP is an integration surface, not a core loop feature — adding it before the loop was stable would have hidden loop bugs behind protocol bugs | ~3 days (see *Next*) |
| **Memory retrieval is literal n-gram, not semantic** | ChromaDB's default ONNX embedding crashes on Windows (onnxruntime access violation), and code-flavoured memory (paths, error strings, function names) is mostly literal repetition anyway | ~0.5 day behind the `MemoryStore` interface, plus an embedding dependency |
| **No process-level sandbox** | Application-level confinement only: path allowlist, dangerous-command rules, Plan mode. A real sandbox means containers or seccomp | 1 week+ to be trustworthy; a fake one is worse than none |
| **No progress detection** | The only brakes are `max_turns` and `max_cost`. Detecting "the model is looping without progress" requires defining *progress* — which takes the decision away from the model, contradicting the loop's core principle | 2–3 days, and it changes the design philosophy |
| **`[[snip:KEY]]` cache is write-only** | Snip is **lossy by design**. Nothing reads the placeholder back; the task list is explicitly excluded so it survives compression | ~1 day to make it readable on demand |
| **No online/deployed data** | Every number in this repo is an offline benchmark. No users, no QPS, no production latency | Not obtainable for a personal project — stated rather than faked |

---

## Next (ordered by value ÷ cost)

### 1. MCP client (stdio, tools only) — ~3 days

**Why**: the built-in tool list does not scale — every new integration is another
class in `core/tools/`. MCP turns "add an integration" into "point at a server".

**Scope**: spawn a stdio server → `initialize` handshake → `tools/list` → wrap each
remote tool as a local `Tool`; `tools/call` on execution; process cleanup on exit.
Deliberately **not** included: SSE transport, `resources`, `prompts`.

**Cost**: protocol framing (`Content-Length` headers, not line-delimited JSON),
tool-name namespacing, timeout/zombie handling on Windows, and a security decision —
remote tools are untrusted, so they should default to the L3/L4 path.

### 2. Semantic memory backend — ~0.5 day

**Why**: n-gram retrieves by surface overlap; "how did I fix an NPE last time"
should match across wording.

**Scope**: one more `MemoryStore` implementation (embeddings + cosine), selected by
config. The interface already exists, so the loop does not change.

**Cost**: a new dependency plus ~500ms/query. Should ship with a measured
comparison against the n-gram baseline, otherwise it is a downgrade with better
branding.

### 3. Readable compaction (replace `[[snip:KEY]]` with a real lookup) — ~1 day

**Why**: Snip currently trades correctness of recall for tokens. A cache that can be
read back keeps the saving and restores the content when the model asks for it.

**Cost**: a tool (`recall_snipped`) plus a policy for when the model is allowed to
expand — otherwise the agent can undo its own compression.

### 4. Progress detection — ~2–3 days

**Why**: a model that keeps calling tools successfully but makes no progress is not
caught today.

**Cost**: defining progress, and a deliberate exception to "the model decides".

### 5. Container sandbox for Bash — 1 week+

**Why**: application-level rules cannot contain a determined command.

**Cost**: Docker/seccomp, image management, and a story for Windows.

---

## Not planned

| Item | Why not |
|---|---|
| GUI / TUI | The value is in the loop, not the rendering surface |
| Multi-language ports | A port multiplies maintenance without exercising new design decisions |
| Framework migration (LangChain/LangGraph) | The point of this repo is that every line is traceable; see the trade-off table in `README.md` |
