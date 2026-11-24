"""Typed lifecycle hooks for the agent runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    USER_PROMPT = "user_prompt"
    AFTER_COMPRESSION = "after_compression"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    STOP = "stop"


HookCallback = Callable[..., Any]


@dataclass(frozen=True)
class HookDecision:
    allow: bool = True
    reason: str = ""


class HookError(RuntimeError):
    """Raised when a runtime hook fails."""

    def __init__(self, event: HookEvent, callback_name: str, cause: Exception):
        self.event = event
        self.callback_name = callback_name
        super().__init__(f"{event.value} hook failed in {callback_name}: {cause}")


class HookRegistry:
    """Run callbacks in registration order for a fixed lifecycle event."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[HookCallback]] = {
            event: [] for event in HookEvent
        }

    def register(self, event: HookEvent, callback: HookCallback) -> HookCallback:
        if not isinstance(event, HookEvent):
            raise TypeError("event must be a HookEvent")
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._hooks[event].append(callback)
        return callback

    async def emit(self, event: HookEvent, *args, **kwargs) -> list[Any]:
        results = []
        for callback in tuple(self._hooks[event]):
            try:
                result = callback(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                callback_name = getattr(
                    callback, "__qualname__", getattr(callback, "__name__", repr(callback))
                )
                raise HookError(event, callback_name, exc) from exc
            results.append(result)
        return results
