# Agent Hook Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a typed, ordered lifecycle hook registry and make tool permissions the first built-in hook consumer.

**Architecture:** A new pure-Python registry owns callback ordering and async invocation. `AgentLoop` triggers events at fixed lifecycle points and treats tool denial or hook failure as fail-closed.

**Tech Stack:** Python 3.11+, asyncio, pytest/pytest-asyncio

---

### Task 1: Hook registry primitives

**Files:**
- Create: `core/hooks.py`
- Create: `tests/test_hooks.py`

- [ ] **Step 1: Write the failing registry tests**

```python
@pytest.mark.asyncio
async def test_hooks_run_in_registration_order_and_support_async():
    registry = HookRegistry()
    seen = []
    registry.register(HookEvent.USER_PROMPT, lambda task: seen.append(("sync", task)))

    async def async_hook(task):
        seen.append(("async", task))

    registry.register(HookEvent.USER_PROMPT, async_hook)
    results = await registry.emit(HookEvent.USER_PROMPT, "fix tests")

    assert seen == [("sync", "fix tests"), ("async", "fix tests")]
    assert results == [None, None]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_hooks.py -q`

Expected: import failure because `core.hooks` does not exist.

- [ ] **Step 3: Implement the minimal registry**

Create `HookEvent`, `HookDecision`, `HookError`, and `HookRegistry.register/emit`. Await awaitable callback results and preserve registration order.

- [ ] **Step 4: Add failing failure/decision tests**

Cover exception wrapping and multiple `BEFORE_TOOL` decisions.

- [ ] **Step 5: Complete the registry behavior and rerun**

Run: `python -m pytest tests/test_hooks.py -q`

Expected: all hook-registry tests pass.

### Task 2: Integrate lifecycle events into AgentLoop

**Files:**
- Modify: `core/agent_loop.py`
- Create: `tests/test_agent_loop_hooks.py`

- [ ] **Step 1: Write a failing lifecycle-order integration test**

Use the existing mock model/tool pattern to record `USER_PROMPT`, `BEFORE_MODEL`, `AFTER_MODEL`, `BEFORE_TOOL`, `AFTER_TOOL`, and `STOP`.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_agent_loop_hooks.py -q`

Expected: `AgentLoop` does not accept `hook_registry`, so the test fails before model execution.

- [ ] **Step 3: Add optional hook registry and trigger observer events**

Trigger prompt, compression, model, tool, and stop events at their documented lifecycle points. Keep event payloads minimal and non-mutating.

- [ ] **Step 4: Run the integration test**

Run: `python -m pytest tests/test_agent_loop_hooks.py -q`

Expected: lifecycle order test passes.

### Task 3: Make permissions a built-in hook

**Files:**
- Modify: `core/agent_loop.py`
- Modify: `tests/test_agent_loop_hooks.py`

- [ ] **Step 1: Write failing tests for hook denial and permission preservation**

Test an external `BEFORE_TOOL` hook denying a call so the tool body never runs, and test that the built-in permission hook still blocks destructive calls.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_agent_loop_hooks.py -q`

Expected: denial is not honored yet.

- [ ] **Step 3: Register the permission adapter as a built-in hook**

Replace the direct permission call in `run_one()` with `HookRegistry.emit(BEFORE_TOOL, ...)` and map any deny decision to a deterministic tool result.

- [ ] **Step 4: Rerun focused tests**

Run: `python -m pytest tests/test_hooks.py tests/test_agent_loop_hooks.py tests/test_hardening.py -q`

Expected: all focused tests pass.

### Task 4: Documentation and full regression

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`

- [ ] **Step 1: Document lifecycle events and failure semantics**

Add a concise section showing registration, event order, and the difference between observer hooks and blocking `BEFORE_TOOL` hooks.

- [ ] **Step 2: Run the complete suite**

Run: `python -m pytest -q`

Expected: all tests pass except the already documented Docker skip when Docker is unavailable.

- [ ] **Step 3: Verify no test-only hook API leaked into production**

Run: `rg -n "HookRegistry|BEFORE_TOOL|HookDecision" core tests README.md DESIGN.md`

Expected: public API appears in `core/hooks.py`, is used by `core/agent_loop.py`, and is covered by tests/docs.
