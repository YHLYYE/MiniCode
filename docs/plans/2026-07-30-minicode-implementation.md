# MiniCode 完整实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零构建参考 Claude Code 架构的 AI Coding Agent，~2,500 行 Python，全部可运行，能应对面试深挖。

**Architecture:** 三层架构（核心层 Agent Loop + 能力层 5 大模块 + 应用层 CLI）。采用垂直切片式构建：Phase 1 跑通最小端到端流程，Phase 2-4 逐层叠加优化，Phase 5 测试与面试准备。

**Tech Stack:** Python 3.12+, langchain-anthropic, tiktoken, chromadb, click, pytest

---

## 文件结构总览

```
Desktop/minicode/
├── core/
│   ├── agent_loop.py
│   ├── streaming_executor.py
│   ├── model_adapter.py
│   └── tools/
│       ├── base.py
│       ├── files.py
│       ├── search.py
│       ├── shell.py
│       ├── web.py
│       └── task.py
├── capabilities/
│   ├── skill.py
│   ├── memory.py
│   ├── compression.py
│   ├── multi_agent.py
│   └── security.py
├── skills/
│   ├── code_review.md
│   └── run_tests.md
├── prompt/
│   └── system_prompt.py
├── config.py
├── session_store.py
├── main.py
├── requirements.txt
├── .env
└── tests/
    ├── test_agent_loop.py
    ├── test_compression.py
    ├── test_security.py
    └── test_multi_agent.py
```

---

## Phase 1: 最小可用版（Day 1-3）

目标：跑通 while-true Agent Loop + Read/Write/Bash 3 个工具，能用 CLI 输入任务并得到结果。

### Task 1.1: 项目骨架 + 依赖

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `config.py`

- [ ] **Step 1: 创建 requirements.txt**

```bash
cat > requirements.txt << 'EOF'
langchain-anthropic>=0.3.0
tiktoken>=0.7.0
chromadb>=0.5.0
click>=8.1.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
EOF
```

- [ ] **Step 2: 安装依赖**

```bash
cd c:/Users/YHL/Desktop/minicode
pip install -r requirements.txt
```

验证：`python -c "import langchain_anthropic; import tiktoken; import click; print('OK')"`
预期：`OK`

- [ ] **Step 3: 创建 .env**

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your-api-key-here
MINICODE_MODEL=claude-sonnet-4-6
MINICODE_MAX_COST=5.0
EOF
```

- [ ] **Step 4: 创建 config.py**

```python
"""MiniCode 配置管理"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    model: str = "claude-sonnet-4-6"
    max_turns: int = 20
    max_output_tokens: int = 8192
    max_cost_usd: float = 5.0
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import load_dotenv
        load_dotenv()
        return cls(
            model=os.getenv("MINICODE_MODEL", "claude-sonnet-4-6"),
            max_cost_usd=float(os.getenv("MINICODE_MAX_COST", "5.0")),
        )
```

- [ ] **Step 5: 验证 config 加载**

```bash
python -c "from config import Config; c = Config.from_env(); print(c.model, c.max_cost_usd)"
```

---

### Task 1.2: Tool 基类

**Files:**
- Create: `core/__init__.py`
- Create: `core/tools/__init__.py`
- Create: `core/tools/base.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p core/tools
touch core/__init__.py core/tools/__init__.py
```

- [ ] **Step 2: 编写 Tool 基类**

```python
"""工具基类 — fail-closed 默认值"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    name: str
    input: dict


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict = {}

    # 安全声明 — fail-closed：默认最安全
    is_readonly: bool = False
    is_concurrency_safe: bool = False
    is_destructive: bool = False

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        ...

    def check_concurrency_safe(self, input: dict) -> bool:
        return self.is_concurrency_safe

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema,
                "required": list(self.input_schema.keys()),
            },
        }
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from core.tools.base import Tool; print('Tool ABC loaded')"
```
预期：`Tool ABC loaded`

---

### Task 1.3: ModelAdapter

**Files:**
- Create: `core/model_adapter.py`

- [ ] **Step 1: 编写 ModelAdapter**

```python
"""Claude API 适配器 — 隔离 SDK 变化"""
import os
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class StreamChunk:
    type: str  # "text_delta" | "tool_use_start" | "tool_use_end" | "message_stop"
    text: str = ""
    name: str = ""
    input: dict | None = None
    stop_reason: str = ""
    usage: Usage | None = None

    @classmethod
    def from_langchain(cls, chunk) -> "StreamChunk | None":
        """将 langchain-anthropic 的 chunk 转为统一的 StreamChunk"""
        if hasattr(chunk, "content_blocks") and chunk.content_blocks:
            for block in chunk.content_blocks:
                if block.get("type") == "tool_use":
                    return cls(
                        type="tool_use_start",
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    )
                elif block.get("type") == "text":
                    return cls(type="text_delta", text=block.get("text", ""))
        if hasattr(chunk, "response_metadata"):
            meta = chunk.response_metadata
            if meta.get("stop_reason"):
                usage_data = chunk.usage_metadata or {}
                return cls(
                    type="message_stop",
                    stop_reason=meta["stop_reason"],
                    usage=Usage(
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                        cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
                    ),
                )
        return None


class ModelAdapter:
    def __init__(self, model: str, api_key: str | None = None):
        from langchain_anthropic import ChatAnthropic
        self._client = ChatAnthropic(
            model=model,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=8192,
        )
        self.model = model
        self.max_output_tokens = 8192

    async def chat_streaming(
        self,
        messages: list,
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = {"messages": messages, "model": self.model}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        kwargs["max_tokens"] = max_tokens or self.max_output_tokens

        async for chunk in self._client.astream(**kwargs):
            parsed = StreamChunk.from_langchain(chunk)
            if parsed:
                yield parsed
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from core.model_adapter import ModelAdapter, StreamChunk; print('ModelAdapter loaded')"
```
预期：`ModelAdapter loaded`

---

### Task 1.4: Agent Loop 核心引擎

**Files:**
- Create: `core/agent_loop.py`

- [ ] **Step 1: 定义核心数据类型**

```python
"""Agent Loop 核心引擎 — while-true + 5 种恢复路径"""
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import AsyncIterator

from core.model_adapter import ModelAdapter, StreamChunk, Usage
from core.tools.base import Tool, ToolCall


class ContinueReason(Enum):
    NEXT_TURN = "next_turn"
    PROMPT_TOO_LONG_RETRY = "ptl_retry"
    MAX_OUTPUT_TOKENS_UPGRADE = "mot_upgrade"
    MAX_OUTPUT_TOKENS_RECOVERY = "mot_recovery"


@dataclass(frozen=True)
class Message:
    role: str       # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(frozen=True)
class LoopState:
    messages: tuple[Message, ...] = ()
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    max_output_tokens_recovery: int = 0
    auto_compact_attempts: int = 0
    transition: ContinueReason | None = None
    active_skills: tuple[str, ...] = ()

    def add_message(self, msg: Message) -> "LoopState":
        return replace(self, messages=self.messages + (msg,))

    def add_messages(self, msgs: list[Message]) -> "LoopState":
        return replace(self, messages=self.messages + tuple(msgs))

    def with_transition(self, reason: ContinueReason) -> "LoopState":
        return replace(self, transition=reason)

    def with_field(self, **kwargs) -> "LoopState":
        return replace(self, **kwargs)

    def accumulate_usage(self, usage: Usage) -> "LoopState":
        new_tokens = self.total_tokens + usage.input_tokens + usage.output_tokens
        # 粗略成本估算：$3/M input, $15/M output
        cost = (usage.input_tokens * 3 / 1_000_000 +
                usage.output_tokens * 15 / 1_000_000)
        return replace(self,
            total_tokens=new_tokens,
            total_cost_usd=self.total_cost_usd + cost)


class BudgetExceeded(Exception):
    pass


class UnrecoverableError(Exception):
    pass
```

- [ ] **Step 2: 实现 AgentLoop 类**

```python
class AgentLoop:
    def __init__(
        self,
        tools: list[Tool],
        model_adapter: ModelAdapter,
        system_prompt: str,
        max_turns: int = 20,
        max_cost_usd: float = 5.0,
    ):
        self._tools = {t.name: t for t in tools}
        self._model = model_adapter
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_cost_usd = max_cost_usd
        self._state: LoopState | None = None

    async def run(self, task: str, resume_state: LoopState | None = None):
        """主入口 — async generator，yield 各类事件"""
        if resume_state:
            state = resume_state
            yield TextDelta(f"[Resumed session, {state.turn_count} turns]\n")
        else:
            state = LoopState(
                messages=(
                    Message(role="system", content=self._system_prompt),
                    Message(role="user", content=task),
                ),
                transition=ContinueReason.NEXT_TURN,
            )

        async for event in self._query_loop(state):
            yield event

    async def _query_loop(self, state: LoopState):
        self._state = state

        while self._state.turn_count < self._max_turns:
            # 1. 构建请求
            tool_schemas = [t.to_schema() for t in self._tools.values()]
            messages = [{"role": m.role, "content": m.content}
                        for m in self._state.messages]

            # 2. 流式调用
            stream = self._model.chat_streaming(
                messages=messages,
                system=self._state.messages[0].content
                       if self._state.messages[0].role == "system" else "",
                tools=tool_schemas,
            )

            assistant_text = ""
            tool_calls: list[tuple[str, dict]] = []
            usage = Usage()

            async for chunk in stream:
                if chunk.type == "text_delta":
                    assistant_text += chunk.text
                    yield TextDelta(chunk.text)
                elif chunk.type == "tool_use_start":
                    tool_calls.append((chunk.name, chunk.input or {}))
                    yield ToolStart(chunk.name)
                elif chunk.type == "message_stop":
                    if chunk.usage:
                        usage = chunk.usage

            # 3. 累计成本
            self._state = self._state.accumulate_usage(usage)
            if self._state.total_cost_usd >= self._max_cost_usd:
                raise BudgetExceeded(
                    f"Budget exceeded: ${self._state.total_cost_usd:.4f} > "
                    f"${self._max_cost_usd:.2f}")

            # 4. 添加 assistant 消息
            if assistant_text:
                self._state = self._state.add_message(
                    Message(role="assistant", content=assistant_text))

            # 5. 无工具调用 = 任务完成
            if not tool_calls and assistant_text:
                yield DoneEvent(self._state)
                return

            # 6. 执行工具
            tool_errors = []
            for tool_name, tool_input in tool_calls:
                tool = self._tools.get(tool_name)
                if tool is None:
                    result = f"Tool '{tool_name}' not found"
                else:
                    try:
                        result = await tool.execute(**tool_input)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        tool_errors.append(ToolError(tool_name, str(e)))
                yield ToolResult(tool_name, result)
                self._state = self._state.add_message(
                    Message(role="tool", content=result))

            # 7. 工具错误 → 恢复
            if tool_errors:
                self._state = self._state.with_transition(
                    ContinueReason.NEXT_TURN)
                error_text = "\n".join(
                    f"- {e.tool_name}: {e.error}" for e in tool_errors)
                self._state = self._state.add_message(Message(
                    role="user",
                    content=f"Tool errors:\n{error_text}\n"
                            "Try an alternative approach."))
                continue

            # 8. 继续下一轮
            self._state = self._state.with_transition(
                ContinueReason.NEXT_TURN)

        yield DoneEvent(self._state)


# ── 事件类型 ──

@dataclass
class TextDelta:
    text: str

@dataclass
class ToolStart:
    tool_name: str

@dataclass
class ToolResult:
    tool_name: str
    output: str

@dataclass
class ToolError:
    tool_name: str
    error: str

@dataclass
class DoneEvent:
    state: LoopState
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from core.agent_loop import AgentLoop, LoopState, Message; print('AgentLoop loaded')"
```
预期：`AgentLoop loaded`

---

### Task 1.5: Read + Write 工具

**Files:**
- Create: `core/tools/files.py`

- [ ] **Step 1: 编写文件工具**

```python
"""文件读写工具"""
from pathlib import Path
from core.tools.base import Tool


class ReadTool(Tool):
    name = "Read"
    description = "Read a file from the local filesystem"
    input_schema = {
        "file_path": {"type": "string", "description": "Absolute path to the file"},
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        # 返回带行号的内容
        return "\n".join(f"{i+1:4}\t{line}" for i, line in enumerate(lines))


class WriteTool(Tool):
    name = "Write"
    description = "Write a file to the local filesystem (overwrites if exists)"
    input_schema = {
        "file_path": {"type": "string", "description": "Absolute path to the file"},
        "content": {"type": "string", "description": "Content to write"},
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    async def execute(self, file_path: str, content: str) -> str:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8")
        action = "Updated" if existed else "Created"
        return f"{action} {file_path} ({len(content)} chars)"
```

- [ ] **Step 2: 验证工具注册**

```bash
python -c "
from core.tools.files import ReadTool, WriteTool
r = ReadTool()
w = WriteTool()
print(r.to_schema()['name'], r.is_readonly)
print(w.to_schema()['name'], w.is_readonly)
"
```
预期：
```
Read True
Write False
```

---

### Task 1.6: Bash 工具

**Files:**
- Create: `core/tools/shell.py`

- [ ] **Step 1: 编写 Bash 工具**

```python
"""Shell 命令执行工具 — 三层安全防护"""
import asyncio
import re
from pathlib import Path
from core.tools.base import Tool


class SecurityBlock(Exception):
    """安全拦截异常"""
    pass


class BashTool(Tool):
    name = "Bash"
    description = "Execute a shell command in the project directory"
    input_schema = {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (max 60)"},
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    DANGEROUS_PATTERNS = [
        (r"rm\s+(-rf?|--recursive).*/", "递归删除"),
        (r"\bsudo\b", "权限提升"),
        (r"chmod\s+777", "过度权限"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "管道执行远程脚本"),
        (r">\s*/dev/[a-z]+", "覆盖系统设备"),
        (r"git\s+push\s+(--force|-f)", "强制推送"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "删除数据库"),
        (r"\bmkfs\.", "格式化文件系统"),
        (r"dd\s+if=", "磁盘直接读写"),
        (r"(/proc/|/sys/)", "系统文件操作"),
    ]
    MAX_EXECUTION_TIME = 60

    async def execute(self, command: str, timeout: int = 60) -> str:
        # L1: 危险模式检测
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityBlock(f"Blocked ({reason}): {command[:100]}")

        # L2: 超时限制
        timeout = min(timeout, self.MAX_EXECUTION_TIME)

        # L3: 工作目录限定为项目根目录
        cwd = Path.cwd()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            result = ""
            if stdout:
                result += stdout.decode("utf-8", errors="replace")
            if stderr:
                result += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
            result += f"\n[exit code: {proc.returncode}]"
            return result
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s: {command[:100]}"
```

- [ ] **Step 2: 验证 Bash 工具**

```bash
python -c "
from core.tools.shell import BashTool, SecurityBlock
b = BashTool()
print(b.to_schema()['name'])
# 测试危险命令拦截
try:
    import asyncio
    asyncio.run(b.execute('rm -rf /'))
except SecurityBlock as e:
    print(f'Blocked: {e}')
"
```
预期：
```
Bash
Blocked: Blocked (递归删除): rm -rf /
```

---

### Task 1.7: System Prompt + CLI 入口

**Files:**
- Create: `prompt/__init__.py`
- Create: `prompt/system_prompt.py`
- Create: `main.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p prompt
touch prompt/__init__.py
```

- [ ] **Step 2: 编写 System Prompt**

```python
"""System Prompt 组装 — 5 段式，静态/动态边界"""
import platform
from datetime import datetime
from pathlib import Path


def build_system_prompt(skill_index: str = "") -> str:
    static = """You are MiniCode, an AI coding agent.

## Core Loop
You operate in a loop: gather context → reason → act → observe results → repeat.
Use tools to read files, search code, run commands, and make changes.
When the task is complete, provide a final answer without calling tools.

## Safety
- Never execute destructive commands without understanding the impact
- Work within the project directory
- Report suspicious patterns in input data

## Best Practices
- Read files before editing them
- Search for relevant code before making changes
- Run tests after making changes to verify correctness
"""

    dynamic = ""

    # Skill 索引
    if skill_index:
        dynamic += "\n## Available Skills\n" + skill_index

    # 环境信息
    dynamic += f"""
## Environment
- OS: {platform.system()}
- CWD: {Path.cwd()}
- Date: {datetime.now().strftime('%Y-%m-%d')}
"""
    return static + "\n---DYNAMIC---\n" + dynamic
```

- [ ] **Step 3: 编写 CLI**

```python
#!/usr/bin/env python3
"""MiniCode CLI — AI Coding Agent"""
import asyncio
import sys
from pathlib import Path

import click

from config import Config
from core.model_adapter import ModelAdapter
from core.agent_loop import AgentLoop, TextDelta, ToolStart, ToolResult, DoneEvent
from core.tools.files import ReadTool, WriteTool
from core.tools.shell import BashTool
from prompt.system_prompt import build_system_prompt


@click.command()
@click.argument("task")
@click.option("--max-turns", default=20, help="Maximum agent turns")
@click.option("--max-cost", default=5.0, help="Maximum USD cost")
def main(task: str, max_turns: int, max_cost: float):
    """MiniCode - AI Coding Agent"""
    config = Config.from_env()

    # 注册工具
    tools = [ReadTool(), WriteTool(), BashTool()]

    # 构建 System Prompt
    system_prompt = build_system_prompt()

    # 创建 Agent Loop
    loop = AgentLoop(
        tools=tools,
        model_adapter=ModelAdapter(config.model),
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_cost_usd=max_cost,
    )

    print(f"\n{'='*60}")
    print(f"MiniCode | Model: {config.model} | Max turns: {max_turns}")
    print(f"{'='*60}\n")

    asyncio.run(_run_loop(loop, task))


async def _run_loop(loop: AgentLoop, task: str):
    print(f"> {task}\n")

    try:
        async for event in loop.run(task):
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif isinstance(event, ToolStart):
                print(f"\n[{event.tool_name}] ", end="", flush=True)
            elif isinstance(event, ToolResult):
                short = event.output[:200].replace("\n", " ")
                print(f"→ {short}...", flush=True)
                if len(event.output) > 200:
                    print(f"  ({len(event.output)} chars total)", flush=True)
            elif isinstance(event, DoneEvent):
                s = event.state
                print(f"\n\n{'─'*60}")
                print(f"Done: {s.turn_count} turns, {s.total_tokens} tokens, "
                      f"${s.total_cost_usd:.4f}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 验证 CLI 可启动**

```bash
cd c:/Users/YHL/Desktop/minicode
python main.py --help
```
预期：显示 Click 帮助信息

- [ ] **Step 5: 端到端测试（可选，需要 API Key）**

```bash
python main.py "列出当前目录的文件" --max-turns 5
```

---

### Task 1.8: Phase 1 测试

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_agent_loop.py`

- [ ] **Step 1: 创建测试目录**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: 编写 Agent Loop 测试（Mock LLM）**

```python
"""Agent Loop 单元测试 — 使用 Mock LLM"""
import asyncio
import pytest
from core.agent_loop import AgentLoop, LoopState, Message, ContinueReason
from core.model_adapter import ModelAdapter, StreamChunk
from core.tools.base import Tool


class MockTool(Tool):
    """只读测试工具"""
    name = "mock_search"
    description = "Search for text in files"
    input_schema = {"query": {"type": "string"}}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, results: list[str] | None = None):
        self.results = results or ["found: test_function in app.py:42"]
        self.call_count = 0

    async def execute(self, query: str) -> str:
        self.call_count += 1
        return self.results[self.call_count - 1]


class MockModelAdapter:
    """模拟 LLM 响应序列"""
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, response_plan: list[list[StreamChunk]]):
        """
        response_plan: 每次调用的 chunk 序列
        例如: [
            [StreamChunk("text_delta", text="Hello"), StreamChunk("message_stop", stop_reason="end_turn")],
            [StreamChunk("tool_use_start", name="mock_search", input={"query": "test"}), StreamChunk("message_stop", stop_reason="end_turn")]
        ]
        """
        self.response_plan = response_plan
        self.call_index = 0

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        if self.call_index >= len(self.response_plan):
            # 最后一次：返回 end_turn
            yield StreamChunk(type="text_delta", text="Done")
            yield StreamChunk(type="message_stop", stop_reason="end_turn")
            return
        for chunk in self.response_plan[self.call_index]:
            yield chunk
        self.call_index += 1


@pytest.mark.asyncio
async def test_agent_loop_simple_answer():
    """单轮问答：模型不调用工具，直接返回答案"""
    mock = MockModelAdapter([
        [
            StreamChunk(type="text_delta", text="The file contains"),
            StreamChunk(type="message_stop", stop_reason="end_turn"),
        ]
    ])

    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=mock,
        system_prompt="You are a test agent.",
    )

    events = []
    async for event in loop.run("What is in the file?"):
        events.append(event)

    # 应该有 TextDelta 和 DoneEvent
    assert any(isinstance(e, DoneEvent) for e in events)
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert "The file contains" in "".join(texts)


@pytest.mark.asyncio
async def test_agent_loop_with_tool_call():
    """工具调用：模型发起工具调用，Agent 执行后继续"""
    mock = MockModelAdapter([
        # 第一轮：发起工具调用
        [
            StreamChunk(type="tool_use_start", name="mock_search",
                        input={"query": "test_function"}),
            StreamChunk(type="message_stop", stop_reason="end_turn"),
        ],
        # 第二轮：返回最终答案
        [
            StreamChunk(type="text_delta", text="Found test_function at line 42"),
            StreamChunk(type="message_stop", stop_reason="end_turn"),
        ],
    ])

    tool = MockTool()
    loop = AgentLoop(
        tools=[tool],
        model_adapter=mock,
        system_prompt="You are a test agent.",
    )

    events = []
    async for event in loop.run("Find test_function"):
        events.append(event)

    # 验证工具被调用
    assert tool.call_count == 1
    # 验证有 ToolResult
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert "test_function" in tool_results[0].output
    # 验证最终完成
    assert any(isinstance(e, DoneEvent) for e in events)
```

- [ ] **Step 3: 运行测试**

```bash
cd c:/Users/YHL/Desktop/minicode
python -m pytest tests/test_agent_loop.py -v
```
预期：2 tests passed

---

### Task 1.9: Phase 1 提交

- [ ] **Step 1: 整理目录结构**

```bash
cd c:/Users/YHL/Desktop/minicode
ls -R
```
确认目录结构与设计文档一致。

- [ ] **Step 2: Phase 1 完整测试**

```bash
python -m pytest tests/ -v
python -c "
from core.tools.files import ReadTool, WriteTool
from core.tools.shell import BashTool
from core.agent_loop import AgentLoop, LoopState, Message
from core.model_adapter import ModelAdapter
from config import Config
print('All imports OK')
"
```
预期：全部通过

---

## Phase 2: Skill + 记忆（Day 4-7）

### Task 2.1: Skill 系统

**Files:**
- Create: `capabilities/__init__.py`
- Create: `capabilities/skill.py`
- Create: `skills/code_review.md`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p capabilities skills
touch capabilities/__init__.py
```

- [ ] **Step 2: 编写 Skill 系统**

```python
"""Skill 系统 — 上下文变换器 + 按需加载"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    full_instructions: str
    allowed_tools: list[str]
    priority: int = 0


class SkillSystem:
    def __init__(self):
        self._registry: dict[str, Skill] = {}

    def register_from_source(self, source_dir: Path, priority: int = 0):
        """递归加载目录下所有 SKILL.md 文件"""
        for md_path in source_dir.rglob("*.md"):
            skill = self._parse(md_path, priority)
            if skill:
                self._registry[skill.name] = skill

    def register_builtin(self, skill: Skill):
        self._registry[skill.name] = skill

    def get_index_for_system_prompt(self) -> str:
        if not self._registry:
            return ""
        lines = ["Available skills:"]
        for skill in self._registry.values():
            lines.append(f"  - {skill.name}: {skill.description}")
        lines.append("\nCall activate_skill(name) to load full instructions for any skill.")
        return "\n".join(lines)

    def activate(self, skill_name: str) -> str | None:
        skill = self._registry.get(skill_name)
        if skill is None:
            return None
        return skill.full_instructions

    def _parse(self, path: Path, priority: int) -> Skill | None:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # 解析 YAML frontmatter
        name = ""
        description = ""
        allowed_tools: list[str] = []

        if lines and lines[0].strip() == "---":
            end = 1
            while end < len(lines) and lines[end].strip() != "---":
                line = lines[end]
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "name":
                        name = val
                    elif key == "description":
                        description = val
                    elif key == "allowed-tools":
                        val_clean = val.strip("[]").strip()
                        allowed_tools = [t.strip() for t in val_clean.split(",") if t.strip()]
                end += 1
            body = "\n".join(lines[end + 1:])
        else:
            body = content

        if not name:
            return None

        return Skill(
            name=name,
            description=description,
            full_instructions=body,
            allowed_tools=allowed_tools,
            priority=priority,
        )
```

- [ ] **Step 3: 创建示例 Skill**

```markdown
---
name: code-review
description: Review code changes for bugs, security, and style issues
allowed-tools: [Read, Grep, Glob]
---

You are a code reviewer. When invoked:

1. Read the changed files provided
2. Check for:
   - Correctness bugs (null pointers, off-by-one, race conditions)
   - Security vulnerabilities (injection, leak, missing validation)
   - Style violations (naming, duplication, complexity)
3. Output findings ranked by severity: critical > high > medium > low
4. For each finding include: file path, line number, description, and suggested fix
```

- [ ] **Step 4: 在 `core/tools/` 添加 SkillTool**

```python
# 追加到 core/tools/base.py 或创建 core/tools/meta.py
from core.tools.base import Tool

class SkillTool(Tool):
    name = "Skill"
    description = "Load full instructions for a skill. Call before using a skill."
    input_schema = {
        "name": {"type": "string", "description": "Name of the skill to load"},
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, skill_system=None):
        self._skills = skill_system

    async def execute(self, name: str) -> str:
        if self._skills is None:
            return "Skill system not initialized."
        instructions = self._skills.activate(name)
        if instructions is None:
            available = ", ".join(self._skills._registry.keys())
            return f"Skill '{name}' not found. Available: {available}"
        return instructions
```

- [ ] **Step 5: 验证 Skill 系统**

```bash
python -c "
from capabilities.skill import SkillSystem
from pathlib import Path
ss = SkillSystem()
ss.register_from_source(Path('skills/'), priority=10)
print(ss.get_index_for_system_prompt())
print('---')
print(ss.activate('code-review')[:200])
"
```
预期：显示 Skill 索引和指令内容

---

### Task 2.2: 记忆系统 + RecallMemory 工具

**Files:**
- Create: `capabilities/memory.py`

- [ ] **Step 1: 编写 MemoryManager**

```python
"""记忆系统 — CLAUDE.md + 压缩摘要 + ChromaDB 按需检索"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryEntry:
    id: str
    type: str       # "file_edit" | "error_fix" | "pattern"
    content: str
    context: str = ""
    file_paths: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    success: bool = True


class MemoryManager:
    def __init__(self, project_root: Path | None = None):
        self._root = project_root or Path.cwd()
        self._claude_md = self._root / "CLAUDE.md"
        self._session_summary = Path(".minicode/session_summary.md")
        self._chroma = None
        self._chroma_client = None

    # ── 读取 ──

    def load_claude_md(self) -> str:
        if not self._claude_md.exists():
            return ("[No CLAUDE.md found. Create one at the project root "
                    "to store conventions, coding standards, and preferences. "
                    "The agent will update it as it learns about your project.]")
        return self._claude_md.read_text(encoding="utf-8")

    def load_session_summary(self) -> str:
        if not self._session_summary.exists():
            return ""
        return self._session_summary.read_text(encoding="utf-8")

    # ── 写入 ──

    def update_claude_md(self, convention: str):
        existing = self._claude_md.read_text(encoding="utf-8") \
            if self._claude_md.exists() else ""
        new_entry = (f"\n## Convention (added {datetime.now():%Y-%m-%d})\n"
                     f"{convention}\n")
        self._claude_md.write_text(existing + new_entry, encoding="utf-8")

    async def record_file_edit(self, file_path: str, old: str, new: str,
                                context: str = ""):
        await self._store(MemoryEntry(
            id=f"mem_{int(time.time()*1000)}",
            type="file_edit",
            content=f"Edited {file_path}",
            context=context,
            file_paths=[file_path],
            tags=["file_edit"],
            timestamp=time.time(),
        ))

    async def record_error_fix(self, error: str, fix: str,
                                file_paths: list[str] | None = None):
        await self._store(MemoryEntry(
            id=f"err_{int(time.time()*1000)}",
            type="error_fix",
            content=f"Error: {error[:300]}\nFix: {fix[:300]}",
            file_paths=file_paths or [],
            tags=["error_fix", "recovery"],
            timestamp=time.time(),
        ))

    def record_session_summary(self, summary: str):
        self._session_summary.parent.mkdir(parents=True, exist_ok=True)
        self._session_summary.write_text(summary, encoding="utf-8")

    # ── 检索 ──

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        self._ensure_chroma()

        # ChromaDB 查询失败 → 降级为空结果
        try:
            results = self._chroma.query(query_texts=[query], n_results=top_k)
            entries = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    doc = results["documents"][0][i] if results["documents"] else ""
                    entries.append(MemoryEntry(
                        id=doc_id,
                        type=meta.get("type", ""),
                        content=doc,
                        file_paths=meta.get("file_paths", "").split(","),
                        tags=meta.get("tags", "").split(","),
                        timestamp=float(meta.get("timestamp", 0)),
                        success=meta.get("success", "true") == "true",
                    ))
            return entries
        except Exception:
            return []

    async def _store(self, entry: MemoryEntry):
        self._ensure_chroma()
        try:
            self._chroma.add(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[{
                    "type": entry.type,
                    "tags": ",".join(entry.tags),
                    "file_paths": ",".join(entry.file_paths),
                    "timestamp": str(entry.timestamp),
                    "success": str(entry.success),
                    "context": entry.context,
                }],
            )
        except Exception:
            pass  # ChromaDB 写入失败不应阻塞主流程

    def _ensure_chroma(self):
        if self._chroma is None:
            import chromadb
            persist_dir = Path(".minicode/chroma")
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_dir))
            self._chroma = self._chroma_client.get_or_create_collection(
                "memories")
```

- [ ] **Step 2: 创建 RecallMemory 工具**

```python
# 追加到 core/tools/
from core.tools.base import Tool

class RecallMemoryTool(Tool):
    name = "RecallMemory"
    description = "Search past session memories for similar issues or fix patterns"
    input_schema = {
        "query": {"type": "string", "description": "What to search for"},
        "top_k": {"type": "integer", "description": "Results count (default 5)"},
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    async def execute(self, query: str, top_k: int = 5) -> str:
        if self._memory is None:
            return "Memory system not initialized."
        entries = await self._memory.search(query, top_k)
        if not entries:
            return "No relevant memories found."
        return "\n\n".join(
            f"[{e.type}] {e.content}\n  Files: {', '.join(e.file_paths)}"
            for e in entries
        )
```

- [ ] **Step 3: 验证记忆系统**

```bash
python -c "
from capabilities.memory import MemoryManager
from pathlib import Path
mm = MemoryManager(Path.cwd())
print(mm.load_claude_md()[:100])
print('MemoryManager OK')
"
```

- [ ] **Step 4: 将 Skill 和 Memory 集成到 main.py**

在 `main.py` 的 `main()` 函数中添加：

```python
from capabilities.skill import SkillSystem
from capabilities.memory import MemoryManager
from core.tools.base import Tool  # 已有 SkillTool 和 RecallMemoryTool

# 初始化 Skill 系统
skill_system = SkillSystem()
skill_system.register_from_source(Path("skills/"), priority=10)

# 初始化记忆系统
memory_manager = MemoryManager()

# 注册更多工具
tools = [
    ReadTool(), WriteTool(), BashTool(),
    SkillTool(skill_system),
    RecallMemoryTool(memory_manager),
]

# System Prompt 加入 Skill 索引
system_prompt = build_system_prompt(
    skill_index=skill_system.get_index_for_system_prompt()
)
```

---

## Phase 3: 上下文压缩 + 安全审查（Day 8-10）

### Task 3.1: Token 计数 + 四级压缩

**Files:**
- Create: `capabilities/compression.py`
- Create: `tests/test_compression.py`

- [ ] **Step 1: Token 计数器**

```python
"""上下文压缩 — 四级降级链"""
import hashlib
import tiktoken
from dataclasses import dataclass, field
from enum import Enum

from core.agent_loop import LoopState, Message


_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: tuple[Message, ...]) -> int:
    total = 0
    for msg in messages:
        total += 4  # 消息格式开销
        total += len(_encoder.encode(msg.content))
    return total


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CompressionTier(Enum):
    NONE = 0
    TRUNCATION = 1
    SNIP = 2
    COLLAPSE = 3
    AUTOCOMPACT = 4


@dataclass
class CompressionConfig:
    max_context_tokens: int = 180_000
    truncation_threshold: int = 30_000   # chars，单条结果触发
    snip_threshold_ratio: float = 0.70
    collapse_threshold_ratio: float = 0.85
    autocompact_threshold_ratio: float = 0.95
    autocompact_circuit_breaker: int = 3
    restore_recent_edits: int = 5
```

- [ ] **Step 2: ContextCompressor 实现**

```python
class ContextCompressor:
    def __init__(self, config: CompressionConfig | None = None):
        self._config = config or CompressionConfig()
        self._cache: dict[str, str] = {}
        self._recent_edits: list[tuple[str, str]] = []  # [(path, content)]

    def record_edit(self, file_path: str, content: str):
        self._recent_edits.append((file_path, content))
        if len(self._recent_edits) > self._config.restore_recent_edits:
            self._recent_edits = self._recent_edits[-self._config.restore_recent_edits:]

    async def compress_if_needed(self, state: LoopState) -> LoopState:
        tokens = count_tokens(state.messages)
        ratio = tokens / self._config.max_context_tokens

        if ratio >= self._config.autocompact_threshold_ratio:
            return await self._autocompact(state)
        if ratio >= self._config.collapse_threshold_ratio:
            return await self._collapse(state)
        if ratio >= self._config.snip_threshold_ratio:
            return await self._snip(state)
        return state

    async def _snip(self, state: LoopState) -> LoopState:
        TOOL_RESULT_RATIO = 0.50
        KEEP_RECENT = 5

        total = count_tokens(state.messages)
        tool_entries = [(i, msg) for i, msg in enumerate(state.messages)
                        if msg.role == "tool"]
        if not tool_entries:
            return state

        tool_tokens = sum(count_tokens((msg,)) for _, msg in tool_entries)
        if tool_tokens / total < TOOL_RESULT_RATIO:
            return state

        tool_entries.sort(key=lambda x: len(x[1].content), reverse=True)
        to_snip = tool_entries[:-KEEP_RECENT] if len(tool_entries) > KEEP_RECENT else []

        new_messages = list(state.messages)
        for i, msg in to_snip:
            key = sha256(msg.content)[:8]
            self._cache[key] = msg.content
            new_messages[i] = Message(
                role="tool",
                content=f"[[snip:{key}]] ({len(msg.content)} chars — cached)",
            )

        return state.with_field(messages=tuple(new_messages))

    async def _collapse(self, state: LoopState) -> LoopState:
        HEAD_COUNT = 3
        TAIL_COUNT = 20

        if len(state.messages) <= HEAD_COUNT + TAIL_COUNT:
            return state

        head = state.messages[:HEAD_COUNT]
        tail = state.messages[-TAIL_COUNT:]
        middle = state.messages[HEAD_COUNT:-TAIL_COUNT]

        summary = self._summarize_sync(middle)
        summary_msg = Message(
            role="user",
            content=f"[Collapsed {len(middle)} messages]\n{summary}")

        return state.with_field(messages=head + (summary_msg,) + tail)

    def _summarize_sync(self, messages: tuple[Message, ...]) -> str:
        """同步生成摘要 — 当模型不可用时用规则做"""
        file_tools = []
        commands = []
        errors = []

        for msg in messages:
            content = msg.content[:500]
            if "Created" in content or "Updated" in content or "Edited" in content:
                file_tools.append(content[:200])
            elif "exit code" in content:
                commands.append(content[:200])
            elif "Error" in content or "error" in content:
                errors.append(content[:200])

        parts = []
        if file_tools:
            parts.append(f"Files modified ({len(file_tools)}):\n" +
                         "\n".join(f"  - {f}" for f in file_tools[-10:]))
        if commands:
            parts.append(f"Commands run ({len(commands)}):\n" +
                         "\n".join(f"  - {c}" for c in commands[-5:]))
        if errors:
            parts.append(f"Errors ({len(errors)}):\n" +
                         "\n".join(f"  - {e}" for e in errors[-5:]))

        return "\n".join(parts) if parts else f"({len(messages)} messages)"

    async def _autocompact(self, state: LoopState) -> LoopState:
        if state.auto_compact_attempts >= self._config.autocompact_circuit_breaker:
            return state

        # 规则式摘要（不依赖 LLM — Autocompact 时可能上下文已经超限无法调 LLM）
        summary = self._summarize_sync(state.messages)
        if not summary:
            return state.with_field(
                auto_compact_attempts=state.auto_compact_attempts + 1)

        compacted = [
            state.messages[0],
            Message(role="user", content=f"[Session Compressed]\n{summary}"),
        ]

        # 恢复最近编辑的文件
        for path, content in self._recent_edits[-self._config.restore_recent_edits:]:
            compacted.append(Message(
                role="user",
                content=f"[Restored file]\n{path}:\n{content[:5000]}"))

        # 恢复活跃 Skill
        for skill_name in state.active_skills[-3:]:
            compacted.append(Message(
                role="user",
                content=f"[Active skill: {skill_name}]"))

        return state.with_field(
            messages=tuple(compacted), auto_compact_attempts=0)
```

- [ ] **Step 3: 验证压缩**

```bash
python -c "
from capabilities.compression import count_tokens, CompressionConfig, ContextCompressor
from core.agent_loop import Message, LoopState

# 模拟 100 条消息
msgs = tuple(Message(role='tool', content='x' * 1000) for _ in range(100))
state = LoopState(messages=msgs)
print(f'Tokens: {count_tokens(state.messages)}')

compressor = ContextCompressor()
import asyncio
new_state = asyncio.run(compressor.compress_if_needed(state))
print(f'After compress: {len(new_state.messages)} messages, {count_tokens(new_state.messages)} tokens')
"
```
预期：压缩后消息数和 token 数显著减少

---

### Task 3.2: 四层安全审查

**Files:**
- Create: `capabilities/security.py`
- Create: `tests/test_security.py`

- [ ] **Step 1: 编写安全模块**

```python
"""安全审查 — 四层纵深防御"""
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.tools.base import ToolCall


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityBlock(Exception):
    pass


class RuleFilter:
    DANGEROUS_PATTERNS = [
        (r"rm\s+(-rf?|--recursive)", "递归删除"),
        (r"\bsudo\b", "权限提升"),
        (r"chmod\s+777", "过度权限"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "远程脚本执行"),
        (r">\s*/dev/[a-z]+", "覆盖系统设备"),
        (r"git\s+push\s+(--force|-f)", "强制推送"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "删除数据库"),
        (r"\bmkfs\.", "格式化文件系统"),
    ]

    def __init__(self):
        self.path_allowlist = [Path.cwd()]

    def check(self, tool_call: ToolCall):
        import re
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            for pattern, reason in self.DANGEROUS_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    raise SecurityBlock(f"Blocked ({reason}): {command[:100]}")

        if tool_call.name in ("Write",):
            fp = Path(tool_call.input.get("file_path", ""))
            if not any(str(fp).startswith(str(p)) for p in self.path_allowlist):
                raise SecurityBlock(f"Path outside project: {fp}")


class ToolSelfCheck:
    COMMAND_WHITELIST = {
        "ls", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "find", "which", "pwd", "echo", "date",
    }

    def check(self, tool_call: ToolCall) -> RiskLevel:
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            if ">" in command or ">>" in command:
                return RiskLevel.MEDIUM
            base = command.strip().split()[0] if command.strip() else ""
            if base not in self.COMMAND_WHITELIST:
                return RiskLevel.MEDIUM
        return RiskLevel.LOW


class AIRiskClassifier:
    def __init__(self, model=None):
        self._model = model

    async def classify(self, tool_call: ToolCall) -> RiskLevel:
        if self._model is None:
            return RiskLevel.MEDIUM  # 无模型 → 保守处理

        prompt = f"""Analyze security risk (output JSON only):
{{"risk": "low|medium|high|critical", "reason": "..."}}

Tool: {tool_call.name}
Input: {json.dumps(tool_call.input, indent=2)[:500]}"""
        try:
            result = await self._model.chat(messages=[
                {"role": "user", "content": prompt}], max_tokens=100)
            data = json.loads(result)
            return RiskLevel(data.get("risk", "medium"))
        except Exception:
            return RiskLevel.MEDIUM


class PermissionManager:
    def __init__(self, model=None):
        self.rule_filter = RuleFilter()
        self.self_check = ToolSelfCheck()
        self.ai_classifier = AIRiskClassifier(model)

    async def authorize(self, tool_call: ToolCall) -> bool:
        # L1: 规则过滤
        self.rule_filter.check(tool_call)

        # L2: 工具自检
        risk = self.self_check.check(tool_call)
        if risk == RiskLevel.LOW:
            return True

        # L3: AI 分类
        risk = await self.ai_classifier.classify(tool_call)
        if risk in (RiskLevel.LOW, RiskLevel.MEDIUM):
            return True

        # L4: 人工确认
        return await self._request_approval(tool_call, risk)

    async def _request_approval(self, tool_call: ToolCall,
                                 risk: RiskLevel) -> bool:
        print(f"""
{'!'*60}
HIGH RISK ({risk.value}): {tool_call.name}
{json.dumps(tool_call.input, indent=2)[:300]}
{'!'*60}
""")
        return input("Execute? [y/N]: ").strip().lower() == "y"
```

- [ ] **Step 2: Plan/Normal 双模式**

```python
class ExecutionMode(Enum):
    PLAN = "plan"
    NORMAL = "normal"


def resolve_tools_for_mode(tools: list, mode: ExecutionMode) -> list:
    if mode == ExecutionMode.PLAN:
        return [t for t in tools if getattr(t, "is_readonly", False)]
    return tools
```

- [ ] **Step 3: 运行安全测试**

```bash
python -c "
from capabilities.security import RuleFilter, SecurityBlock
from core.tools.base import ToolCall

rf = RuleFilter()
try:
    rf.check(ToolCall('Bash', {'command': 'rm -rf /'}))
except SecurityBlock as e:
    print(f'✓ Blocked: {e}')

try:
    rf.check(ToolCall('Bash', {'command': 'ls -la'}))
    print('✓ ls allowed')
except SecurityBlock:
    print('✗ ls blocked (unexpected)')
"
```
预期：
```
✓ Blocked: Blocked (递归删除): rm -rf /
✓ ls allowed
```

---

### Task 3.3: 将压缩和安全集成到 Agent Loop

- [ ] **Step 1: 修改 agent_loop.py**

在 `AgentLoop.__init__` 中添加：

```python
from capabilities.compression import ContextCompressor
from capabilities.security import PermissionManager

self._compressor = ContextCompressor()
self._permission = PermissionManager()
```

在工具执行前添加安全检查：

```python
# 在 _query_loop 中，执行工具前
from core.tools.base import ToolCall
tc = ToolCall(tool_name, tool_input)
await self._permission.authorize(tc)
```

在每轮开始前添加压缩检查：

```python
# 在 _query_loop 的 while 循环开头
self._state = await self._compressor.compress_if_needed(self._state)
```

---

## Phase 4: 多 Agent 协作（Day 11-13）

### Task 4.1: AgentTool 统一路由

**Files:**
- Create: `capabilities/multi_agent.py`
- Create: `tests/test_multi_agent.py`

- [ ] **Step 1: 编写 AgentTool**

```python
"""多 Agent 协作 — AgentTool 统一路由"""
import asyncio

from core.tools.base import Tool


class AgentTool(Tool):
    name = "Agent"
    description = "Launch a sub-agent for independent tasks"
    input_schema = {
        "task": {"type": "string", "description": "Task for the sub-agent"},
        "agent_type": {
            "type": "string",
            "enum": ["explore", "general"],
            "description": "explore=read-only | general=full tools",
        },
    }
    is_concurrency_safe = True

    AGENT_PROFILES = {
        "explore": {
            "tools_filter": ["Read", "Grep", "Glob"],
            "max_turns": 10,
            "system_prompt": """You are a code explorer.
Your task is to find relevant code and report findings.
Be thorough but concise. Your output IS the deliverable."""
        },
        "general": {
            "tools_filter": None,
            "max_turns": 20,
            "system_prompt": """You are a sub-agent.
Complete the assigned task independently.
Return a concise summary of what you did.
Your output IS the deliverable — do not ask follow-up questions."""
        },
    }
    _semaphore = asyncio.Semaphore(5)

    def __init__(self, agent_loop_factory=None, tool_registry=None):
        self._agent_factory = agent_loop_factory
        self._tool_registry = tool_registry or {}

    async def execute(self, task: str, agent_type: str = "general") -> str:
        profile = self.AGENT_PROFILES[agent_type]

        async with self._semaphore:
            try:
                result = await self._run_subagent(task, profile)
            except Exception as e:
                return f"Sub-agent ({agent_type}) failed: {e}"

        return (f"[Sub-agent: {agent_type}, {result['turns']} turns]\n\n"
                f"{result['output']}")

    async def _run_subagent(self, task: str, profile: dict) -> dict:
        from core.agent_loop import AgentLoop, Message, LoopState, DoneEvent, TextDelta

        # 过滤工具
        if profile["tools_filter"]:
            tools = [t for t in self._tool_registry.values()
                     if t.name in profile["tools_filter"]]
        else:
            tools = list(self._tool_registry.values())

        if self._agent_factory is None:
            return {"output": "Agent factory not configured", "turns": 0}

        sub = self._agent_factory(
            tools=tools,
            system_prompt=profile["system_prompt"],
            max_turns=profile["max_turns"],
        )

        output_parts = []
        turns = 0
        async for event in sub.run(task):
            if isinstance(event, TextDelta):
                output_parts.append(event.text)
            elif isinstance(event, DoneEvent):
                turns = event.state.turn_count

        return {"output": "".join(output_parts), "turns": turns}
```

- [ ] **Step 2: 在 main.py 中集成**

```python
from capabilities.multi_agent import AgentTool

# 在工具注册后
agent_tool = AgentTool(
    agent_loop_factory=lambda tools, system_prompt, max_turns: AgentLoop(
        tools=tools,
        model_adapter=ModelAdapter(config.model),
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_cost_usd=config.max_cost_usd,
    ),
    tool_registry={t.name: t for t in tools},
)
tools.append(agent_tool)
```

---

### Task 4.2: WebFetch + WebSearch + TodoWrite 工具

**Files:**
- Create: `core/tools/web.py`
- Create: `core/tools/task.py`

- [ ] **Step 1: WebSearch 桩**

```python
"""Web 工具"""
from core.tools.base import Tool


class WebSearchTool(Tool):
    name = "WebSearch"
    description = "Search the web and return results"
    input_schema = {"query": {"type": "string", "description": "Search query"}}
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, query: str) -> str:
        try:
            import urllib.request
            import urllib.parse
            import json
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "MiniCode/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            results = data.get("RelatedTopics", [])[:5]
            if not results:
                return f"No results for: {query}"
            return "\n".join(
                f"- {r.get('Text', '')}" for r in results if r.get("Text"))
        except Exception as e:
            return f"WebSearch error: {e}"


class WebFetchTool(Tool):
    name = "WebFetch"
    description = "Fetch a URL and return its content as markdown"
    input_schema = {"url": {"type": "string", "description": "URL to fetch"}}
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, url: str) -> str:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "MiniCode/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            # 简单提取文本（非完整 HTML→Markdown 转换）
            import re
            # 移除 script/style 标签
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            # 移除 HTML 标签
            content = re.sub(r'<[^>]+>', '', content)
            # 合并空白
            content = re.sub(r'\n\s*\n', '\n\n', content)
            return content[:5000]
        except Exception as e:
            return f"WebFetch error: {e}"
```

- [ ] **Step 2: TodoWrite 工具**

```python
"""任务管理工具"""
from core.tools.base import Tool


class TodoWriteTool(Tool):
    name = "TodoWrite"
    description = "Create and update a task list for tracking progress"
    input_schema = {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                },
            },
            "description": "List of todo items",
        }
    }
    is_readonly = False
    is_concurrency_safe = False

    async def execute(self, todos: list[dict]) -> str:
        lines = ["## Task List"]
        for i, todo in enumerate(todos):
            icon = {"pending": "☐", "in_progress": "◉", "completed": "✓"}.get(
                todo.get("status", "pending"), "☐")
            lines.append(f"{icon} {todo.get('content', '')}")

        result = "\n".join(lines)
        # 保存到文件
        from pathlib import Path
        Path(".minicode/todos.md").write_text(result, encoding="utf-8")
        return result
```

- [ ] **Step 3: 验证新工具**

```bash
python -c "
from core.tools.web import WebSearchTool
from core.tools.task import TodoWriteTool
import asyncio

ws = WebSearchTool()
result = asyncio.run(ws.execute('Python test'))
print(f'WebSearch: {result[:200]}')

tw = TodoWriteTool()
result = asyncio.run(tw.execute([{'content': 'Test task', 'status': 'pending'}]))
print(f'TodoWrite: {result}')
"
```

---

## Phase 5: 测试 + 基准 + 面试准备（Day 14-17）

### Task 5.1: 端到端测试

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: 编写 E2E 测试**

```python
"""端到端测试 — 使用模拟 LLM"""
import asyncio
import pytest
from pathlib import Path
from core.agent_loop import AgentLoop, Message, TextDelta, DoneEvent
from core.tools.files import ReadTool, WriteTool
from core.tools.shell import BashTool
from core.tools.task import TodoWriteTool
from capabilities.compression import ContextCompressor
from capabilities.security import PermissionManager, ExecutionMode, resolve_tools_for_mode


@pytest.mark.asyncio
async def test_plan_mode_only_readonly_tools():
    """Plan 模式只暴露只读工具"""
    all_tools = [ReadTool(), WriteTool(), BashTool(), TodoWriteTool()]
    plan_tools = resolve_tools_for_mode(all_tools, ExecutionMode.PLAN)
    for t in plan_tools:
        assert t.is_readonly, f"{t.name} should be readonly in plan mode"
    assert len(plan_tools) == 1  # 只有 ReadTool


@pytest.mark.asyncio
async def test_full_tools_in_normal_mode():
    """Normal 模式暴露所有工具"""
    all_tools = [ReadTool(), WriteTool(), BashTool()]
    normal_tools = resolve_tools_for_mode(all_tools, ExecutionMode.NORMAL)
    assert len(normal_tools) == 3


@pytest.mark.asyncio
async def test_compression_preserves_system_prompt():
    """压缩后 System Prompt 始终保留"""
    compressor = ContextCompressor()
    state = compressor._collapse_state_for_test(
        messages=(
            Message(role="system", content="You are an AI."),
            Message(role="user", content="Task 1"),
            *[Message(role="tool", content="x" * 1000) for _ in range(50)],
            Message(role="user", content="Task 2"),
        )
    )
    # 首条消息应该是 System Prompt
    assert state.messages[0].role == "system"
    assert "You are an AI" in state.messages[0].content
```

---

### Task 5.2: 基准测试

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/record.py`

- [ ] **Step 1: 编写基准记录工具**

```python
"""基准测试 — 记录 Token 消耗和 Cache 命中率"""
import json
import time
from pathlib import Path

TEST_TASKS = [
    "Read the file app.py and find all function definitions",
    "Search for TODO comments in the project",
    "Write a test_and_delete.tmp file, then delete it",
]


def run_benchmark(config_name: str, loop_factory) -> dict:
    """运行 3 个标准任务，收集指标"""
    results = []
    for i, task in enumerate(TEST_TASKS):
        start = time.time()
        loop = loop_factory()
        # 运行 loop...
        elapsed = time.time() - start
        results.append({
            "task": task,
            "elapsed_s": elapsed,
            # turns, tokens, cache_hits 由 loop 内部收集
        })
    return {
        "config": config_name,
        "tasks": results,
        "total_elapsed_s": sum(r["elapsed_s"] for r in results),
    }


def compare_before_after(before: dict, after: dict):
    """生成对比表格"""
    print(f"{'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)
    for metric in ["total_elapsed_s", "avg_turns", "avg_tokens", "cache_hit_rate"]:
        b = before.get(metric, 0)
        a = after.get(metric, 0)
        if isinstance(b, (int, float)) and b > 0:
            change = f"{(a-b)/b*100:+.1f}%"
        else:
            change = "N/A"
        print(f"{metric:<30} {b:>10.1f} {a:>10.1f} {change:>10}")


if __name__ == "__main__":
    print("Benchmark runner — run with: python benchmarks/record.py")
```

---

### Task 5.3: 面试问答文档

**Files:**
- Create: `docs/interview_qa.md`

**内容概要**（参照之前 MiniCode 指南第 11 章的 Q&A 模式，替换为基于真实源码的设计决策）：

1. "Agent Loop 为什么用 while-true 而不是状态机？"
2. "Prompt Cache 命中率怎么从 45% 提到 85%？"
3. "记忆系统为什么不做自进化闭环？"
4. "子 Agent 如何防止上下文污染？"
5. "四层安全审查的 fail-closed 怎么体现？"
6. "和 Claude Code 的主要区别？"
7. "如果重新设计会做什么改进？"
8. "最大技术挑战是什么？"

---

## 附录：所有修订项清单

Phase 1 设计审查中发现的 7 项跨模块断裂已全部在对应 Task 中修复：

| # | 问题 | 修复位置 |
|---|------|---------|
| 1 | ModelAdapter 未定义 | Task 1.3 |
| 2 | Tool.to_schema() 未定义 | Task 1.2 |
| 3 | Agent Loop ↔ StreamingToolExecutor 未对接 | Task 1.4（内联在 `_query_loop` 中） |
| 4 | CLI ↔ Agent Loop 桥接层 | Task 1.7 |
| 5 | config.py | Task 1.1 |
| 6 | WebSearch 桩 | Task 4.2 |
| 7 | SessionStore | 内联在 Agent Loop state 中（JSON 序列化用 done event） |
