import pytest

from core.hooks import HookError, HookEvent, HookRegistry


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


@pytest.mark.asyncio
async def test_hook_error_names_event_and_callback():
    registry = HookRegistry()

    async def broken(task):
        raise RuntimeError("boom")

    registry.register(HookEvent.USER_PROMPT, broken)

    with pytest.raises(HookError) as exc_info:
        await registry.emit(HookEvent.USER_PROMPT, "task")

    assert exc_info.value.event is HookEvent.USER_PROMPT
    assert "broken" in exc_info.value.callback_name
    assert isinstance(exc_info.value.__cause__, RuntimeError)
