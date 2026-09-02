# MiniCode 完整设计文档

> 参考 Claude Code 真实源码架构（基于 how-claude-code-works 21 章分析 + claude-code-from-scratch 23 层渐进构建）
> 目标：2-3 周完成，~2,500 行 Python，全部可运行，全部可解释

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心引擎：Agent Loop](#2-核心引擎agent-loop)
3. [工具系统](#3-工具系统)
4. [Skill 系统](#4-skill-系统)
5. [上下文压缩](#5-上下文压缩)
6. [记忆系统](#6-记忆系统)
7. [多 Agent 协作](#7-多-agent-协作)
8. [安全审查](#8-安全审查)
9. [系统组装](#9-系统组装)
10. [逐层构建路线](#10-逐层构建路线)

---

## 1. 整体架构

### 1.1 三层架构

```
┌─────────────────────────────────────────┐
│         应用层（CLI + 配置）               │
│  main.py — Click CLI                     │
│  SessionStore — JSON 对话持久化           │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         能力层（5 大模块）                 │
│  Skill 系统 — 上下文变换器 + 按需加载      │
│  记忆系统 — CLAUDE.md + 摘要 + n-gram 检索 │
│  上下文压缩 — 四级降级链                   │
│  多Agent协作 — AgentTool 统一路由          │
│  安全审查 — 四层纵深防御 + Plan/Normal     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         核心层                            │
│  Agent Loop — while-true + 5 恢复路径     │
│  StreamingToolExecutor — 流式并行执行      │
│  ModelAdapter — Claude API 适配           │
│  Tool Registry — 13 个工具                │
└─────────────────────────────────────────┘
```

### 1.2 目录结构

```
Desktop/minicode/
├── core/
│   ├── agent_loop.py          # while-true + State + 5 恢复路径
│   ├── state.py               # LoopState/Message/事件类型
│   ├── model_adapter.py       # litellm 多模型适配（100+ 模型）
│   └── tools/
│       ├── base.py            # Tool ABC + fail-closed 默认值 + Skill/RecallMemory 工具
│       ├── files.py           # Read, Write（通过 FilesystemBackend）
│       ├── fs_backend.py      # 可插拔文件系统后端（本地/沙箱）
│       ├── shell.py           # Bash（23+ 正则 + 白名单 + 超时）
│       ├── web.py             # WebFetch, WebSearch
│       └── task.py            # TodoWrite
├── capabilities/
│   ├── skill.py               # Skill 系统（渐进式披露 + 二阶段路由）
│   ├── memory.py              # 三类记忆 + 可插拔 MemoryStore
│   ├── compression.py         # 四级降级链
│   ├── multi_agent.py         # AgentTool 统一路由（explore/general/worktree/team）
│   └── security.py            # 四层纵深防御
├── skills/                    # Skill 定义文件
│   └── code_review.md
├── prompt/
│   └── system_prompt.py       # 5 段式 System Prompt
├── benchmarks/                # 基准测试（实测量化数据）
│   ├── record.py              # 四级压缩比
│   └── e2e.py                 # 端到端 Token 成本对比
├── config.py                  # 配置管理
├── session_store.py           # 对话持久化（--resume）
├── main.py                    # CLI 入口（交互式 REPL）
├── requirements.txt
└── tests/                     # 47 个测试
    ├── test_agent_loop.py     # Agent Loop + 恢复路径
    ├── test_skill.py          # Skill 路由
    ├── test_memory.py         # 三类记忆
    ├── test_multi_agent.py    # 多 Agent 协作
    ├── test_security.py       # 安全审查
    ├── test_recovery.py       # 恢复路径
    ├── test_fs_backend.py     # 文件系统后端
    └── test_session_store.py  # 会话持久化
```

### 1.3 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| LLM 调用 | `litellm`（统一 100+ 模型） | 借鉴 mini-swe-agent，模型无关；一个接口支持 Claude/DeepSeek/OpenAI/Gemini/本地 |
| Agent Loop | 100% 自研 while-true | 面试核心，逐行可解释 |
| 工具装饰器 | 自研 Tool ABC | 控制所有行为，不依赖框架 |
| CLI | Click | 简单，`python main.py` 交互式 REPL |
| 记忆检索 | 纯 Python n-gram 哈希 + 余弦相似度 | 评估 ChromaDB 后（Windows onnxruntime 崩溃）自研，零外部模型 |
| 记忆存储 | 可插拔 `MemoryStore` 接口（默认 JSON） | 借鉴 DeepAgents，可切换 SQLite/ChromaDB/Redis |
| 文件系统 | 可插拔 `FilesystemBackend` 接口（默认本地） | 借鉴 DeepAgents，可切换 Docker 沙箱/远程 |
| 会话持久化 | SessionStore（JSON 文件） | 跨进程恢复，`--resume` 续聊 |
| Token 计数 | tiktoken（cl100k_base） | 精确，Claude/GPT 通用 |

### 1.4 设计哲学（对齐 Claude Code）

1. **模型是唯一决策者** — 不预设状态机，不硬编码路由
2. **可组合性** — 子 Agent 复用主 Agent 同一个 engine
3. **渐进式披露** — Skill 索引常驻，完整指令按需加载
4. **fail-closed** — `is_concurrency_safe=False`、`is_readonly=False` 默认最安全
5. **错误扣留** — 可恢复错误不暴露给用户，静默恢复
6. **架构随模型进步而瘦身** — 每层可独立开关，不互相污染

---

## 2. 核心引擎：Agent Loop

### 2.1 双层架构

```
QueryEngine（外层：会话生命周期）
  ├── max_turns, max_cost_usd 管控
  ├── SessionStore 持久化
  └── 调用 →
        query()（内层：while-true 单轮执行）
          ├── 四级压缩检查
          ├── 构建 API 请求
          ├── 流式调用 + StreamingToolExecutor
          ├── 工具结果回流
          └── 5 种恢复路径 → continue/exit
```

### 2.2 State 对象（不可变）

```python
@dataclass(frozen=True)
class LoopState:
    messages: tuple[Message, ...]       # 完整对话历史
    turn_count: int                     # 当前轮次
    total_tokens: int                   # 累计 token 消耗
    total_cost_usd: float               # 累计美元成本
    max_output_tokens_recovery: int     # output token 升级次数
    auto_compact_attempts: int          # autocompact 重试计数
    transition: ContinueReason | None   # 本轮继续原因
    active_skills: tuple[str, ...]      # 当前活跃的 Skill 名称
```

### 2.3 5 种恢复路径

```python
class ContinueReason(Enum):
    NEXT_TURN = "next_turn"
    PROMPT_TOO_LONG_RETRY = "ptl_retry"
    MAX_OUTPUT_TOKENS_UPGRADE = "mot_upgrade"
    MAX_OUTPUT_TOKENS_RECOVERY = "mot_recovery"
    COMPACT_FAILURE_RETRY = "compact_retry"
```

### 2.4 核心实现

```python
class AgentLoop:
    def __init__(self, tools, model_adapter, system_prompt, max_turns, max_cost_usd):
        self._tools = {t.name: t for t in tools}
        self._model = model_adapter
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_cost_usd = max_cost_usd
        self._compressor = ContextCompressor()
        self._permission = PermissionManager()
        self._executor = StreamingToolExecutor(self._tools)

    async def run(self, task: str):
        state = LoopState(
            messages=(
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=task),
            ),
            turn_count=0,
            total_tokens=0,
            total_cost_usd=0.0,
            max_output_tokens_recovery=0,
            auto_compact_attempts=0,
            transition=ContinueReason.NEXT_TURN,
            active_skills=(),
        )

        async for event in self._query_loop(state):
            yield event

    async def _query_loop(self, state: LoopState):
        while state.turn_count < self._max_turns:
            # 1. 四级压缩检查
            state = await self._compressor.compress_if_needed(state)

            # 2. 构建请求
            request = build_request(state.messages, self._tools, self._model)

            # 3. 流式调用 + 实时工具执行
            stream = self._model.chat_streaming(request)
            tool_results = []
            assistant_content = []

            async for event in self._executor.process_stream(stream):
                if isinstance(event, TextDelta):
                    assistant_content.append(event.text)
                    yield event
                elif isinstance(event, ToolResult):
                    # 安全检查
                    await self._permission.authorize(event.tool_call)
                    tool_results.append(event)
                elif isinstance(event, ToolError):
                    tool_results.append(event)

            # 4. 结果回流
            assistant_msg = Message(role="assistant", content="".join(assistant_content))
            tool_msgs = [Message(role="tool", content=str(r)) for r in tool_results]
            state = state.add_messages([assistant_msg] + tool_msgs)

            # 5. 累计成本
            state = state.accumulate_usage(request.usage)
            if state.total_cost_usd >= self._max_cost_usd:
                raise BudgetExceeded(state.total_cost_usd, self._max_cost_usd)

            # 6. 退出检查
            if request.stop_reason == "end_turn":
                yield DoneEvent(state)
                return

            # 7. 恢复路径
            if request.stop_reason == "max_tokens":
                state = await self._handle_max_tokens(state)
                continue

            if tool_results and any(isinstance(r, ToolError) for r in tool_results):
                errors = [r for r in tool_results if isinstance(r, ToolError)]
                state = await self._handle_tool_errors(state, errors)
                continue

            state = state.with_transition(ContinueReason.NEXT_TURN)

    async def _handle_max_tokens(self, state: LoopState) -> LoopState:
        ESCALATED_MAX_TOKENS = 64000
        MAX_MOT_RETRIES = 3

        if self._model.max_output_tokens < ESCALATED_MAX_TOKENS:
            self._model.max_output_tokens = ESCALATED_MAX_TOKENS
            return state.with_transition(ContinueReason.MAX_OUTPUT_TOKENS_UPGRADE)

        if state.max_output_tokens_recovery < MAX_MOT_RETRIES:
            msg = Message(role="user",
                content="Output token limit hit. Resume directly — no apology, "
                        "no recap. Pick up mid-thought if cut off. "
                        "Break remaining work into smaller pieces.")
            return state.add_message(msg).with_field(
                "max_output_tokens_recovery", state.max_output_tokens_recovery + 1)

        raise UnrecoverableError("max_output_tokens recovery exhausted")

    async def _handle_tool_errors(self, state: LoopState,
                                   errors: list[ToolError]) -> LoopState:
        error_summary = "\n".join(f"- {e.tool_name}: {e.error}" for e in errors)
        msg = Message(role="user",
            content=f"Tool calls failed:\n{error_summary}\n\n"
                    "Analyze the errors and try an alternative. "
                    "Do NOT retry the exact same call.")
        return state.add_message(msg)
```

### 2.5 StreamingToolExecutor

```python
class StreamingToolExecutor:
    """模型还在流式输出时，解析完 tool_use 块就立即执行"""

    def __init__(self, tools: dict[str, Tool]):
        self._tools = tools
        self._pending: list[asyncio.Task] = []

    async def process_stream(self, stream: AsyncIterator[StreamChunk]):
        current_batch: list[ToolContext] = []

        async for chunk in stream:
            if chunk.type == "text_delta":
                yield TextDelta(chunk.text)

            elif chunk.type == "tool_use_start":
                tool = self._tools[chunk.name]
                ctx = ToolContext(
                    tool_name=chunk.name,
                    tool_input=chunk.input,
                    is_concurrency_safe=tool.check_concurrency_safe(chunk.input),
                )

                if ctx.is_concurrency_safe:
                    ctx.task = asyncio.create_task(tool.execute(**chunk.input))
                    current_batch.append(ctx)
                else:
                    await self._drain(current_batch)
                    try:
                        result = await tool.execute(**chunk.input)
                        yield ToolResult(chunk.name, result)
                    except Exception as e:
                        yield ToolError(chunk.name, e)
                    current_batch = []

        await self._drain(current_batch)

    async def _drain(self, batch: list[ToolContext]):
        if not batch:
            return
        results = await asyncio.gather(
            *[ctx.task for ctx in batch], return_exceptions=True)
        for ctx, result in zip(batch, results):
            if isinstance(result, Exception):
                yield ToolError(ctx.tool_name, result)
            else:
                yield ToolResult(ctx.tool_name, result)

    def has_pending(self) -> bool:
        return any(not t.done() for t in self._pending)

    async def drain_all(self):
        await asyncio.gather(*self._pending, return_exceptions=True)
        self._pending.clear()
```

### 2.6 ModelAdapter（litellm 多模型适配）

```python
class ModelAdapter:
    """统一 LLM 接口 — 基于 litellm，支持 100+ 模型。

    借鉴 mini-swe-agent 的模型无关设计。Agent Loop 只看到干净的
    StreamChunk 事件，从不接触原始 provider chunk。换模型是一行配置
    变更，provider 协议差异隐藏在 litellm 内部。
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = self._resolve_model(model)  # "deepseek-chat" → "deepseek/deepseek-chat"
        self.max_output_tokens = 8192

    def _resolve_model(self, model: str) -> str:
        """把 'deepseek-chat' 归一化为 litellm 的 'provider/model' 格式"""
        if "/" in model:
            return model
        provider = os.environ.get("MINICODE_PROVIDER", "deepseek")
        return f"{provider}/{model}"

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        import litellm
        kwargs = {
            "model": self.model, "messages": input_msgs,
            "max_tokens": ..., "stream": True,
            "stream_options": {"include_usage": True},  # 拿到 token 用量
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)  # Anthropic → OpenAI 格式
        # 统一解析 OpenAI 格式流式 chunk → StreamChunk（text_delta / tool_use_start）
        ...
```

> **设计价值**：Anthropic 的 `tool_use` block 和 OpenAI 的 `tool_calls` + `tool_call_id` 关联是两套完全不同的协议。litellm 在适配层归一化，Agent Loop 完全不感知 provider 差异——这是"隔离变化"原则的落地。

---

## 3. 工具系统

### 3.1 Tool 基类（fail-closed 默认值）

```python
class Tool(ABC):
    name: str
    description: str
    input_schema: dict  # JSON Schema properties

    # 安全声明 — 默认最安全
    is_readonly: bool = False
    is_concurrency_safe: bool = False
    is_destructive: bool = False

    @abstractmethod
    async def execute(self, **kwargs) -> str: ...

    def check_concurrency_safe(self, input: dict) -> bool:
        """运行时动态判断"""
        return self.is_concurrency_safe

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema,
                "required": list(self.input_schema.keys()),
            }
        }
```

### 3.2 9 个内置工具

| 工具 | 只读 | 并发安全 | 破坏性 | 职责 |
|------|------|---------|--------|------|
| Read | ✅ | ✅ | - | 读文件（通过 FilesystemBackend） |
| Write | - | - | ✅ | 写文件（通过 FilesystemBackend） |
| Bash | - | - | ✅ | Shell 命令（23+ 正则 + 白名单 + 超时） |
| WebFetch | ✅ | ✅ | - | 获取网页内容 |
| WebSearch | ✅ | ✅ | - | 搜索引擎 |
| TodoWrite | - | - | - | 任务列表管理 |
| Agent | - | ✅ | - | 派发子 Agent（explore/general/worktree/team） |
| Skill | ✅ | ✅ | - | 加载 Skill 指令 |
| RecallMemory | ✅ | ✅ | - | 搜索历史记忆 |

### 3.3 工具执行安全包装

```python
MAX_RESULT_CHARS = 30_000

async def execute_tool_safely(tool: Tool, input: dict,
                               permission: PermissionManager) -> str:
    # 1. 权限检查
    await permission.authorize(ToolCall(tool.name, input))

    # 2. 执行
    result = await tool.execute(**input)

    # 3. 截断超长结果
    if len(result) > MAX_RESULT_CHARS:
        cache_path = Path(f".minicode/cache/{sha256(result)}.txt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result, encoding="utf-8")
        result = (f"[Truncated: {len(result)} chars → {cache_path}]\n"
                  f"[Preview]\n{result[:1000]}\n..."
                  f"({len(result)-1500} chars omitted)...\n{result[-500:]}")
    return result
```

### 3.4 Bash 工具三层安全

```python
class BashTool(Tool):
    name = "Bash"
    DANGEROUS_PATTERNS = [
        (r"rm\s+(-rf?|--recursive).*/", "递归删除根目录"),
        (r"(sudo|chmod\s+777|chown)", "权限提升"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "管道执行远程脚本"),
        (r">\s*/dev/[a-z]+", "覆盖系统设备"),
        (r"git\s+push\s+(--force|-f)", "强制推送"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "删除数据库"),
        (r"mkfs\.", "格式化文件系统"),
        (r"dd\s+if=", "磁盘直接读写"),
        (r"(/proc/|/sys/)", "系统文件操作"),
    ]
    COMMAND_WHITELIST = {
        "ls", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "find", "which", "pwd", "echo", "date",
    }
    MAX_EXECUTION_TIME = 60  # 秒

    async def execute(self, command: str, timeout: int = 60) -> str:
        # L1: 危险模式检测
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityBlock(f"Blocked: {reason}")

        # L2: 超时限制
        timeout = min(timeout, self.MAX_EXECUTION_TIME)

        # L3: 工作目录限定（由 PermissionManager 的路径白名单保证）
        proc = await asyncio.create_subprocess_shell(
            command, stdout=PIPE, stderr=PIPE, cwd=Path.cwd())
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)

        return (stdout.decode("utf-8", errors="replace") +
                stderr.decode("utf-8", errors="replace"))
```

---

## 4. Skill 系统

### 4.1 核心设计：上下文变换器 + 按需加载

```
System Prompt（极简索引，永不变化 → Prompt Cache 命中）
  │
  │ 模型调用 activate_skill("code_review")
  ▼
Tool Result（追加到消息尾部）
  │
  │ <skill name="code_review">
  │ You are a code reviewer...
  │ Allowed tools: Read, Bash, WebSearch
  │ </skill>
  │
  │ → System Prompt 一个字不变
  │ → Prefix Cache 100% 命中
```

### 4.2 Skill 定义格式

```markdown
---
name: code-review
description: Review code changes for bugs, security, and style issues
allowed-tools: [Read, Bash, WebSearch]
user-invocable: true
---

You are a code reviewer. When invoked:
1. Read the changed files
2. Check for: correctness bugs, security vulnerabilities, style violations
3. Output findings ranked by severity (critical > high > medium > low)
4. For each finding: file path, line number, description, fix suggestion
```

### 4.3 实现

```python
class SkillSystem:
    def __init__(self):
        self._registry: dict[str, Skill] = {}

    def register_from_source(self, source_dir: Path, priority: int):
        for skill_md in source_dir.rglob("SKILL.md"):
            skill = self._parse(skill_md, priority)
            self._registry[skill.name] = skill

    def get_index_for_system_prompt(self) -> str:
        lines = ["Available skills:"]
        for skill in self._registry.values():
            lines.append(f"- {skill.name}: {skill.description}")
        lines.append("\nCall activate_skill(name) to load full instructions.")
        return "\n".join(lines)

    async def activate(self, skill_name: str) -> str:
        skill = self._registry.get(skill_name)
        if not skill:
            available = ", ".join(self._registry.keys())
            return f"Skill '{skill_name}' not found. Available: {available}"
        return skill.full_instructions

    def resolve(self, name: str) -> Skill | None:
        candidates = [s for s in self._registry.values() if s.name == name]
        if not candidates:
            return None
        candidates.sort(key=lambda s: s.priority, reverse=True)
        return candidates[0]
```

### 4.4 Skill 加载工具

```python
class SkillTool(Tool):
    name = "Skill"
    description = "Load full instructions for a skill. Call this before using a skill."
    input_schema = {"name": {"type": "string", "description": "Skill name"}}
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, name: str) -> str:
        return await skill_system.activate(name)
```

### 4.5 压缩后重新激活

压缩吞掉 Skill 指令后，恢复最近 3 个活跃 Skill：

```python
# 在 Autocompact 恢复阶段
active_skills = state.active_skills[-3:]
for skill_name in active_skills:
    skill_content = await skill_system.activate(skill_name)
    compacted.append(Message(role="user", content=skill_content))
```

---

## 5. 上下文压缩

### 5.1 四级降级链

```
Token 使用率
  95% → Autocompact：fork 子 Agent 全量摘要
  85% → Collapse：掐头去尾 + 中间结构化摘要
  70% → Snip：旧工具输出 → 占位符替换
  50% → Truncation：单条结果超过 30K chars 时实时截断
```

### 5.2 配置

```python
@dataclass
class CompressionConfig:
    max_context_tokens: int = 180_000
    truncation_threshold: int = 30_000
    snip_threshold_ratio: float = 0.70
    collapse_threshold_ratio: float = 0.85
    autocompact_threshold_ratio: float = 0.95
    autocompact_circuit_breaker: int = 3
    restore_recent_edits: int = 5
```

### 5.3 逐级实现

```python
class ContextCompressor:
    def __init__(self, config: CompressionConfig, model: ModelAdapter):
        self._config = config
        self._model = model
        self._cache: dict[str, str] = {}  # FileCacheStore

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
        """工具输出占比 > 50% → 从最大的开始替换为占位符"""
        TOOL_RESULT_RATIO = 0.50
        KEEP_RECENT = 5

        total = count_tokens(state.messages)
        tool_indices = [(i, msg) for i, msg in enumerate(state.messages)
                        if msg.role == "tool"]
        tool_tokens = sum(count_tokens(msg) for _, msg in tool_indices)

        if tool_tokens / total < TOOL_RESULT_RATIO:
            return state

        tool_indices.sort(key=lambda x: len(x[1].content), reverse=True)
        to_snip = tool_indices[:-KEEP_RECENT] if len(tool_indices) > KEEP_RECENT else []

        new_messages = list(state.messages)
        for i, msg in to_snip:
            cache_key = sha256(msg.content)
            self._cache[cache_key] = msg.content
            new_messages[i] = Message(role="tool",
                content=f"[[snip:{cache_key[:8]}]] ({len(msg.content)} chars)")

        return state.with_messages(tuple(new_messages))

    async def _collapse(self, state: LoopState) -> LoopState:
        """掐头去尾（3+20）+ 中间结构化摘要"""
        HEAD_COUNT = 3
        TAIL_COUNT = 20

        if len(state.messages) <= HEAD_COUNT + TAIL_COUNT:
            return state

        head = state.messages[:HEAD_COUNT]
        tail = state.messages[-TAIL_COUNT:]
        middle = state.messages[HEAD_COUNT:-TAIL_COUNT]

        summary = await self._summarize(middle)
        msg = Message(role="user",
            content=f"[Context Collapse — {len(middle)} messages summarized]\n{summary}")
        return state.with_messages(head + (msg,) + tail)

    async def _autocompact(self, state: LoopState) -> LoopState:
        """Fork 子 Agent 全量摘要 + 熔断器"""
        if state.auto_compact_attempts >= self._config.autocompact_circuit_breaker:
            return state  # 熔断

        summary = await self._fork_summarizer(state.messages)
        if summary is None:
            return state.with_field("auto_compact_attempts",
                                     state.auto_compact_attempts + 1)

        compacted = [
            state.messages[0],  # System Prompt
            Message(role="user", content=f"[Session Compressed]\n{summary}"),
        ]

        # 恢复最近编辑的文件
        for f in self._get_recent_edits(state, self._config.restore_recent_edits):
            compacted.append(Message(role="user",
                content=f"[Restored file]\n{f.path}:\n{f.content}"))

        # 恢复活跃 Skill
        for skill_name in state.active_skills[-3:]:
            skill_content = await skill_system.activate(skill_name)
            compacted.append(Message(role="user", content=skill_content))

        return state.with_messages(tuple(compacted)).with_field(
            "auto_compact_attempts", 0)

    async def _fork_summarizer(self, messages) -> str | None:
        prompt = """You are a conversation archivist. Summarize this agent session:

Output format:
## Task Summary
[What the user asked for]

## Completed Work
[What was accomplished, key decisions]

## Files Modified
| File | Change |
|------|--------|

## Errors & Resolutions

## Current State
[What's in progress, what should happen next]

## Key Context
[Facts the agent needs to continue: paths, names, decisions]

Keep under 2000 words. Prioritize information to CONTINUE working."""

        try:
            result = await self._model.chat(
                messages=[Message(role="user", content=prompt),
                          Message(role="user", content=format_messages(messages))],
                max_tokens=8000)
            return result
        except Exception:
            return None

    async def _summarize(self, messages) -> str:
        prompt = f"""Summarize {len(messages)} messages:
1. Completed tasks + key decisions
2. Files modified
3. Key findings
4. Remaining work

Conversation:
{format_messages(messages)}"""
        return await self._model.chat(prompt, max_tokens=4000)
```

### 5.4 Token 精确计数

```python
import tiktoken
_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages) -> int:
    total = 0
    for msg in messages:
        total += 4  # 消息格式开销
        if isinstance(msg.content, str):
            total += len(_encoder.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                total += len(_encoder.encode(str(block)))
    return total
```

---

## 6. 记忆系统

### 6.1 设计：三类记忆分级存储 + 可插拔后端

```
三类记忆（按认知科学分类）：
  1. 程序性记忆 Procedural — "怎么做"，长期，JSON 文件 + 关键词检索
  2. 情景记忆 Episodic — "发生过什么"，中期，JSON + n-gram 向量检索
  3. 用户画像 User Profile — "用户偏好"，长期，JSON 键值对

存储层可插拔（借鉴 DeepAgents 的 pluggable store backends）：
  MemoryStore 接口 → JSONMemoryStore（默认，零依赖）| SQLite | ChromaDB | Redis
```

### 6.2 实现：可插拔 MemoryStore

```python
class MemoryStore(ABC):
    """可插拔记忆存储后端 — 换存储介质不改 MemoryManager API"""
    @abstractmethod
    def add(self, entry: MemoryEntry) -> None: ...
    @abstractmethod
    def search(self, memory_type, query, top_k) -> list[MemoryEntry]: ...
    @abstractmethod
    def get_profile(self, key) -> str | None: ...
    @abstractmethod
    def set_profile(self, key, value) -> None: ...


class JSONMemoryStore(MemoryStore):
    """默认后端：JSON 文件 + n-gram 向量检索，零外部依赖"""
    def add(self, entry):
        # procedural/episodic → 追加 JSON 列表；profile → 键值对
        ...
    def search(self, memory_type, query, top_k):
        # procedural → 关键词重叠；episodic → n-gram 余弦相似度
        ...


class MemoryManager:
    """高层 API — 会话级上下文（CLAUDE.md/摘要）+ 委托 store 管记忆条目"""
    def __init__(self, project_root=None, store: MemoryStore | None = None):
        self._store = store or JSONMemoryStore(project_root / ".minicode" / "memories")
```

> **工程决策**：原本计划用 ChromaDB 做情景记忆语义检索，但评估发现其默认 ONNX embedding 在 Windows 加载 onnxruntime 时崩溃（access violation）。改为自研纯 Python n-gram 哈希向量 + 余弦相似度——零外部模型、跨平台、保留"向量检索"架构。n-gram 是字面相似度（非语义），对代码领域够用（错误模式/文件路径/函数名都是字面重复）；语义检索是明确扩展点。

### 6.3 RecallMemory 工具

```python
class RecallMemoryTool(Tool):
    name = "RecallMemory"
    description = "Search past session memories for similar issues or patterns"
    input_schema = {
        "query": {"type": "string", "description": "What to search for"},
        "top_k": {"type": "integer", "description": "Number of results (default 5)"},
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, query: str, top_k: int = 5) -> str:
        entries = await memory_manager.search(query, top_k)
        if not entries:
            return "No relevant memories found."
        return "\n\n".join(
            f"[{e.type}] {e.content}\nFiles: {', '.join(e.file_paths)}"
            for e in entries)
```

---

## 7. 多 Agent 协作

### 7.1 核心理念

> 子 Agent 就是同一个 turn engine，换一组参数再跑一遍。

### 7.2 AgentTool 统一路由

```python
class AgentTool(Tool):
    name = "Agent"
    description = "Launch a sub-agent to handle complex, independent tasks"
    input_schema = {
        "task": {"type": "string", "description": "Task for the sub-agent"},
        "agent_type": {
            "type": "string",
            "enum": ["explore", "general", "worktree", "team"],
            "description": "explore=只读搜索 | general=全部工具 | worktree=git隔离 | team=多角色并行",
        },
    }
    is_concurrency_safe = True
    _active_semaphore = asyncio.Semaphore(5)  # 并发上限

    AGENT_PROFILES = {
        "explore": {
            "tools": ["Read", "WebSearch", "WebFetch"],
            "max_turns": 10,
            "system_prompt": "You are a code explorer. Find relevant code. "
                             "Your output IS the deliverable — be thorough but brief.",
        },
        "general": {
            "tools": None,  # None = 继承全部工具
            "max_turns": 20,
            "system_prompt": "Complete the assigned task independently. "
                             "Return a concise summary. "
                             "Your output IS the deliverable.",
        },
    }

    async def execute(self, task: str, agent_type: str = "general",
                      model: str | None = None) -> str:
        profile = self.AGENT_PROFILES[agent_type]

        async with self._active_semaphore:
            sub = AgentLoop(
                tools=self._resolve_tools(profile["tools"]),
                model_adapter=ModelAdapter(model or profile["model"]),
                system_prompt=profile["system_prompt"],
                max_turns=profile["max_turns"],
                max_cost_usd=1.0,
            )

            try:
                result = await sub.run(task)
            except Exception as e:
                return f"Sub-agent ({agent_type}) failed: {e}"

        # 只返回结果摘要（不回传完整上下文）
        return (f"[Sub-agent: {agent_type}, {result.turn_count} turns, "
                f"{result.total_tokens} tokens]\n\n{result.output}")

    def _resolve_tools(self, tool_names: list[str] | None) -> list[Tool]:
        if tool_names is None:
            return list(tool_registry.values())
        return [tool_registry[name] for name in tool_names]
```

### 7.3 上下文隔离

子 Agent 的 `run()` 和主 Agent 是同一个 `AgentLoop` 类，区别只在参数：
- 独立的 `messages[]`（不共享主 Agent 历史）
- 独立的 `system_prompt`
- 独立的 `max_turns` 和 `context_budget`
- 执行完毕后只返回结果摘要到主 Agent 上下文

---

## 8. 安全审查

### 8.1 四层纵深防御

```
工具调用请求
  ↓
① 规则过滤器（<1ms）→ 23+ 危险模式 + 路径白名单 → 拒绝
  ↓ 通过
② 工具自检（<5ms）→ Bash 白名单 / 输出重定向检查
  ↓ 通过
③ AI 风险分类器（~500ms）→ 检测 Prompt 注入 / 范围越界
  ↓ high/critical
④ 人工确认（阻塞，等用户）
  ↓ 确认/拒绝
```

### 8.2 实现

```python
class SecurityBlock(Exception):
    """安全拦截 — 上层捕获后向模型注入拦截信息"""
    pass

class RuleFilter:
    DANGEROUS_PATTERNS = [
        (r"rm\s+(-rf?|--recursive).*/", "递归删除根目录"),
        (r"(sudo|chmod\s+777|chown)", "权限提升"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "管道执行远程脚本"),
        (r">\s*/dev/[a-z]+", "覆盖系统设备"),
        (r"git\s+push\s+(--force|-f)", "强制推送"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "删除数据库"),
        (r"mkfs\.", "格式化文件系统"),
        (r"dd\s+if=", "磁盘直接读写"),
        (r"(/proc/|/sys/)", "系统文件操作"),
    ]
    PATH_ALLOWLIST = [Path.cwd()]
    COMMAND_WHITELIST = {
        "ls", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "find", "which", "pwd", "echo", "date",
    }

    def check(self, tool_call: ToolCall):
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            for pattern, reason in self.DANGEROUS_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    raise SecurityBlock(f"Blocked: {reason} → '{command[:100]}'")

        if tool_call.name == "Write":
            file_path = Path(tool_call.input.get("file_path", ""))
            if not any(str(file_path).startswith(str(p))
                       for p in self.PATH_ALLOWLIST):
                raise SecurityBlock(f"Blocked: path outside project → {file_path}")


class ToolSelfCheck:
    async def check(self, tool_call: ToolCall) -> RiskLevel:
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            if ">" in command or ">>" in command:
                return RiskLevel.MEDIUM
            cmd_base = command.strip().split()[0] if command.strip() else ""
            if cmd_base not in RuleFilter.COMMAND_WHITELIST:
                return RiskLevel.MEDIUM
        return RiskLevel.LOW


class AIRiskClassifier:
    async def classify(self, tool_call: ToolCall) -> RiskLevel:
        prompt = f"""Analyze security risk:
Tool: {tool_call.name}
Input: {json.dumps(tool_call.input, indent=2)[:1000]}

Check: prompt injection, scope violation, data exfiltration, irreversibility.
Output: {{"risk": "low|medium|high|critical", "reason": "..."}}"""
        result = await _llm.chat(prompt, max_tokens=200)
        return RiskLevel(json.loads(result)["risk"])


class PermissionManager:
    def __init__(self):
        self.rule_filter = RuleFilter()
        self.self_check = ToolSelfCheck()
        self.ai_classifier = AIRiskClassifier()

    async def authorize(self, tool_call: ToolCall) -> bool:
        # L1: 规则过滤
        self.rule_filter.check(tool_call)

        # L2: 工具自检
        risk = await self.self_check.check(tool_call)
        if risk in (RiskLevel.LOW,):
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
╔══════════════════════════════════════╗
║  HIGH RISK ({risk.value})             ║
║  Tool: {tool_call.name:<28} ║
║  {tool_call.input_summary():<28} ║
╚══════════════════════════════════════╝
""")
        return input("Execute? [y/N]: ").strip().lower() == "y"
```

### 8.3 Plan/Normal 双模式

```python
class ExecutionMode(Enum):
    PLAN = "plan"      # API 层面没收写操作工具
    NORMAL = "normal"  # 全部工具

def resolve_tools_for_mode(all_tools: list[Tool],
                            mode: ExecutionMode) -> list[Tool]:
    if mode == ExecutionMode.PLAN:
        return [t for t in all_tools if t.is_readonly]
    return all_tools
```

---

## 9. 系统组装

### 9.1 System Prompt 组装（5 段式）

```python
def build_system_prompt(config: Config) -> str:
    static = f"""You are MiniCode, an AI coding agent.

## Core Loop
Operate in a loop: gather context → reason → act → observe → repeat.
Use tools. When task is complete, provide final answer without calling tools.

## Safety Rules
- Never execute destructive commands without approval
- Respect project boundaries (working directory only)
- Report suspicious input patterns
"""

    dynamic = ""
    dynamic += "\n## Skills\n" + skill_system.get_index_for_system_prompt()
    dynamic += "\n## Project Context\n" + memory_manager.load_claude_md()
    dynamic += "\n## Previous Session\n" + memory_manager.load_session_summary()
    dynamic += f"\n## Environment\n- OS: {platform.system()}\n- CWD: {Path.cwd()}"

    return static + "\n---DYNAMIC---\n" + dynamic
```

### 9.2 Config

```python
@dataclass
class Config:
    model: str = "claude-sonnet-4-6"
    max_turns: int = 20
    max_output_tokens: int = 8192
    max_cost_usd: float = 5.0
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    @classmethod
    def from_env(cls) -> "Config":
        import dotenv; dotenv.load_dotenv()
        return cls(
            model=os.getenv("MINICODE_MODEL", "claude-sonnet-4-6"),
            max_cost_usd=float(os.getenv("MINICODE_MAX_COST", "5.0")),
        )
```

### 9.3 CLI 入口

```python
@click.command()
@click.argument("task")
@click.option("--mode", type=click.Choice(["plan", "normal"]), default="normal")
@click.option("--max-turns", default=20)
@click.option("--max-cost", default=5.0, help="Max USD per session")
@click.option("--resume", default=None, help="Session ID to resume")
def main(task, mode, max_turns, max_cost, resume):
    config = Config.from_env()
    config.max_turns = max_turns
    config.max_cost_usd = max_cost
    config.execution_mode = ExecutionMode[mode.upper()]

    # 加载 Skill
    skill_system.register_from_source(Path("skills/"), priority=10)

    # 解析工具
    tools = resolve_tools_for_mode(all_tools, config.execution_mode)

    # 构建 System Prompt
    system_prompt = build_system_prompt(config)

    # 启动
    loop = AgentLoop(
        tools=tools,
        model_adapter=ModelAdapter(config.model),
        system_prompt=system_prompt,
        max_turns=config.max_turns,
        max_cost_usd=config.max_cost_usd,
    )

    asyncio.run(_run_interactive(loop, task, resume))

async def _run_interactive(loop: AgentLoop, task: str, resume: str | None):
    if resume:
        state = SessionStore().load(resume)
        if state is None:
            print(f"Session {resume} not found")
            return
    else:
        state = None

    session_store = SessionStore()
    async for event in loop.run(task, state):
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text); sys.stdout.flush()
        elif isinstance(event, ToolStart):
            print(f"\n[{event.tool_name}] ", end="", flush=True)
        elif isinstance(event, ToolResult):
            print(f"→ {len(event.output)} chars", flush=True)
        elif isinstance(event, DoneEvent):
            print(f"\n\nDone: {event.state.turn_count} turns, "
                  f"{event.state.total_tokens} tokens, "
                  f"${event.state.total_cost_usd:.4f}")

    session_store.save(loop._state)
```

### 9.4 SessionStore

```python
class SessionStore:
    def __init__(self, path: Path = Path(".minicode/sessions/")):
        self._dir = path
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: LoopState) -> str:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"{sid}.json"
        path.write_text(json.dumps({
            "messages": [{"role": m.role, "content": m.content}
                         for m in state.messages],
            "turn_count": state.turn_count,
            "total_tokens": state.total_tokens,
            "total_cost_usd": state.total_cost_usd,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return sid

    def load(self, sid: str) -> LoopState | None:
        path = self._dir / f"{sid}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return LoopState(
            messages=tuple(Message(**m) for m in d["messages"]),
            turn_count=d["turn_count"],
            total_tokens=d.get("total_tokens", 0),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            max_output_tokens_recovery=0,
            auto_compact_attempts=0,
            transition=ContinueReason.NEXT_TURN,
            active_skills=(),
        )
```

---

## 10. 逐层构建路线

### Phase 1 (Day 1-3): 最小可用版 ~480 行

```
目标: 跑通 Agent Loop + 3 个工具 (Read/Write/Bash)
- core/agent_loop.py        — while-true + State + 5 恢复路径
- core/tools/base.py         — Tool ABC
- core/tools/files.py        — Read, Write
- core/tools/shell.py        — Bash（23+ 正则 + 白名单）
- core/model_adapter.py      — Claude API 适配
- core/streaming_executor.py  — 流式并行执行
- prompt/system_prompt.py    — System Prompt 组装
- config.py                  — Config + 环境变量
- main.py                    — CLI 入口
```

### Phase 2 (Day 4-7): Skill + 记忆 ~350 行

```
目标: Skill 按需加载 + CLAUDE.md 读写 + 错误记忆
- capabilities/skill.py      — SkillSystem + SkillTool（含二阶段路由）
- capabilities/memory.py     — MemoryManager + MemoryStore（三类记忆）
- skills/code_review.md      — 示例 Skill
```

### Phase 3 (Day 8-10): 压缩 + 安全 ~500 行

```
目标: 四级降级链 + 四层审查 + Plan/Normal
- capabilities/compression.py — ContextCompressor + 四级降级
- capabilities/security.py    — 四层防御 + 双模式（含 AI 风险分类）
- 完善 Bash 工具 (超时 + 三层安全)
```

### Phase 4 (Day 11-13): 多 Agent ~300 行

```
目标: 多 Agent 协作 + Web/TodoWrite 工具
- capabilities/multi_agent.py — AgentTool 统一路由（explore/general/worktree/team）
- core/tools/web.py           — WebFetch, WebSearch
- core/tools/task.py          — TodoWrite
```

### Phase 5 (Day 14-17): 测试 + 基准 + 面试 ~400 行

```
目标: 端到端测试 + 性能数据 + 面试问答
- tests/test_agent_loop.py
- tests/test_compression.py
- tests/test_security.py
- tests/test_multi_agent.py
- benchmarks/record.py        — 3 个标准任务 + Token/Cache 对比
- docs/interview_qa.md        — 面试问答准备
```

---

## 附录 A: 与 Claude Code 的差异对照

| 维度 | Claude Code 源码 | MiniCode | 差异理由 |
|------|-----------------|----------|---------|
| 代码量 | 512,000 行 | ~2,500 行 | 教育实现，非产品 |
| Agent Loop | while-true + 7 恢复路径 | while-true + 5 恢复路径 | 去掉 stop_hook 和 token_budget_continuation（Python SDK 不适用） |
| 工具数 | 55+ | 13 | 覆盖核心场景，超出范围的不做 |
| Bash 安全 | tree-sitter AST + 23+ 检查 | 23+ 正则 + 白名单 | tree-sitter 是独立项目级复杂度 |
| Skill 加载 | 6 层来源 + 分区排序 | 3 层来源 + 全量索引 | 对 13 个工具不需要分区 |
| 记忆系统 | CLAUDE.md + Hook + 压缩摘要 | CLAUDE.md + 压缩摘要 + n-gram 向量检索 | 自研 n-gram（ChromaDB 在 Windows 崩溃） |
| 子 Agent 协作 | SendMessage + JSONL邮箱 + FSM | AgentTool 统一路由 | 单机不需要对等通信协议 |
| 权限 | 7 层 + AST 分析 | 4 层 | 不需要 iOS/Android 沙箱、OS 级防护 |
| Worktree 隔离 | Git worktree 并行 | worktree 模式（git 隔离，非 git 降级） | 简化版实现 |
| UI | React + Ink（终端框架） | Click CLI + print | 不做终端 UI 框架 |

## 附录 B: 面试核心论述

1. **"为什么 Agent Loop 用 while-true 而不是状态机？"**
   → 模型是唯一决策者。状态机的状态转换是人预设的，while-true 让模型通过 tool_use/end_turn/max_tokens 自主判断。7 种恢复路径不是状态机，是容错策略。

2. **"Prompt Cache 命中率怎么从 45% 提到 85%？"**
   → System Prompt 静态段永不变化（Skill 索引一行一个、不加完整指令），动态段放会话信息。Skill 指令通过 activate_skill 工具返回，作为 Tool Response 追加到消息尾部。静态/动态边界精确定义。

3. **"记忆系统为什么不做自进化闭环？"**
   → 模型参数冻结的情况下，真正的"学习"不可能。我们选择务实方案：CLAUDE.md（文件持久化）+ Autocompact 摘要（压缩存留）+ n-gram 向量检索（按需）。评估 ChromaDB 后发现其默认 ONNX embedding 在 Windows 崩溃，故自研纯 Python n-gram 哈希向量 + 余弦相似度。承认技术边界本身展示工程判断力。

4. **"子 Agent 如何防止上下文污染？"**
   → 子 Agent 复用同一个 AgentLoop 但拥有独立 messages[]。执行完毕只返回结果摘要，不回传完整上下文。AgentTool 统一路由——模型只需要学会用"Agent"这个工具。

5. **"四层安全审查的 fail-closed 怎么体现？"**
   → Tool 默认 is_concurrency_safe=False，忘记声明就串行；默认 is_readonly=False，触发权限检查。L1 正则拒绝危险命令，L2 检查参数合法性，L3 AI 分类，L4 人工确认。Plan 模式 API 层面物理隔离写操作工具。

---

## 附录 C: 性能基准（实测数据，可复现）

两个基准脚本，零成本可重复（不消耗真实 API）：

```bash
python benchmarks/record.py   # 四级压缩比
python benchmarks/e2e.py      # 端到端 Token 成本对比
```

### C.1 压缩比（record.py）

模拟 200 文件长会话（58,030 token）：

| 级别 | 消息数 | Token 数 | 压缩比 |
|------|--------|---------|--------|
| 原始（无压缩） | 402 | 58,030 | — |
| Snip（占位替换） | 402 | 8,110 | **86%** |
| Collapse（掐头去尾） | 24 | 2,959 | **95%** |
| Autocompact（全量） | 2 | 28 | **99.95%** |

### C.2 端到端成本（e2e.py）

模拟读 30 文件长任务，对比开/关压缩的累计 Token：

| 指标 | 关压缩 | 开压缩 |
|------|--------|--------|
| 累计 input token | 1,173,075 | 296,058 |
| 最终上下文 token | 75,657 | 13,376 |

**结论**：
- **Token 成本降低 74.8%**（877K token）
- 关压缩时上下文涨至 75K **超过 DeepSeek 64K 窗口而失败**，压缩使任务得以完成

> **诚实边界**：以上是「模拟端到端」（mock 模型行为 + 真实 token 计数 + 真实压缩逻辑），非真实 API 调用。压缩比是压缩算法的实测，端到端成本是累计 token 的模拟对比。
