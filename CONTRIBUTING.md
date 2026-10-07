# Contributing

Thanks for looking. This is a **personal project** (one author) — it is not trying
to become a framework. That shapes what a good contribution looks like here:
**small, justified, and traceable**.

Before sending anything, please read `README.md` (design trade-offs) and
`ROADMAP.md` (what is deliberately out of scope).

---

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                             # then fill in a model key
```

`MINICODE_MODEL` defaults to `claude-sonnet-4-6`; `deepseek-chat` is the cheapest
path (see `.env.example`).

## Run it

```bash
python main.py --help                 # CLI surface
python main.py "读一下 core/state.py 并总结"     # one-shot task
python main.py                        # interactive REPL
python main.py --mode plan "重构 auth 模块"     # read-only planning mode
python main.py --resume <session_id>  # resume a saved session
```

## Tests and gates

All three must pass before a change is worth reviewing:

```bash
python -m pytest tests/ -q              # 254 tests
python -m pyflakes $(git ls-files '*.py')   # undefined names / unused imports
python benchmarks/verify_claims.py      # 14 documented claims, re-measured
```

`verify_claims.py` is the important one: it re-runs the benchmarks and compares the
result against the numbers written in `README.md` / `DESIGN.md`. **If your change
moves a documented number, update the document — do not delete the claim.**

## How to add a tool

1. Subclass `Tool` in `core/tools/` and implement `execute()`.
2. Declare the three safety-relevant attributes explicitly:

   | Attribute | Default | Meaning |
   |---|---|---|
   | `is_readonly` | `False` | `True` only if it cannot modify anything |
   | `is_concurrency_safe` | `False` | `True` only if it is safe to run in parallel with others |
   | `is_destructive` | `False` | `True` if it writes/executes; forces the L3+L4 security path |

   Leaving them at their defaults is safe — the defaults are the strict ones.
3. Add a `name` and an `input_schema` (used for the tool-calling schema).
4. Register it in `main.py::_build_tools()`.
5. Add a test that exercises **the tool body**, not just its rules. (A missing
   `import re` once shipped because no test ever called `BashTool.execute()`.)

## Conventions

- **Fail closed.** When a check cannot decide, deny. `capabilities/security.py`
  returns HIGH on every classifier failure path for exactly this reason.
- **Inject errors, don't raise them at the model.** A tool failure becomes a tool
  result the model can read and react to; only unrecoverable conditions escape.
- **State is immutable.** `LoopState`/`Message` are frozen dataclasses — use
  `with_field()` / `add_message()`, never mutate.
- **One implementation per rule.** Duplicated logic drifts: two copies of the
  dangerous-command list had already diverged, and two keyword tokenizers only one
  of which handled Chinese.
- Comments explain **why**, and may be written in Chinese (the author's first
  language); code identifiers stay in English.

## PR checklist

- [ ] `pytest`, `pyflakes`, `verify_claims` all pass
- [ ] New behaviour has a test that **fails without your change**
- [ ] No new dead code (unused imports / unreachable branches)
- [ ] If a documented number moved, the document was updated
- [ ] If a boundary was crossed, `ROADMAP.md` was updated

## What will not be merged

- A framework wrapper that hides the loop ("just use LangGraph for this part")
- Features that trade safety for convenience (auto-approving write tools, skipping
  the human path "because it is slow")
- Big refactors without a measured reason — file count is not the metric here
