# Agent 生命周期钩子注册层设计

## Goal

让 MiniCode 的运行时能力通过固定生命周期钩子接入主循环，而不是把所有集成逻辑继续堆进 `AgentLoop._query_loop()`。

这次只实现第一条“运行时内核钩子化”，不实现命令语法树解析，也不改设计文档截图里的第二条安全能力。

## Scope

实现：

- 新增 `core/hooks.py`，提供事件类型、钩子注册、顺序触发、同步/异步回调支持。
- `AgentLoop` 接受可选 `HookRegistry`，在任务开始、模型前后、工具前后、压缩后和停止时触发事件。
- 权限检查注册为内置 `BEFORE_TOOL` 钩子，确保钩子层有真实消费者。
- 前置工具钩子可以返回拒绝决定，拒绝时工具不执行，结果回灌给模型。
- 新增单元测试和主循环集成测试。

不做：

- 动态插件发现、配置文件加载和第三方包入口点。
- 全局钩子单例。
- 任意钩子改写工具参数或模型消息。
- 命令抽象语法树解析。

## Architecture

`HookRegistry` 只负责保存和顺序执行某个事件下的回调，不认识 Agent 状态。`AgentLoop` 决定事件在哪里触发、传什么参数，以及如何处理回调结果。

事件集合：

- `USER_PROMPT`：任务开始，参数为任务文本。
- `AFTER_COMPRESSION`：每次上下文压缩检查后，参数为压缩前后状态。
- `BEFORE_MODEL`：模型请求前，参数为当前状态、消息列表和工具声明。
- `AFTER_MODEL`：一次模型流结束后，参数为状态、文本、工具调用、用量和停止原因。
- `BEFORE_TOOL`：工具执行前，参数为 `ToolCall` 和工具对象；返回 `HookDecision` 可拒绝执行。
- `AFTER_TOOL`：工具成功或失败结果生成后，参数为 `ToolCall`、工具对象和结果字符串。
- `STOP`：产出 `DoneEvent` 前，参数为最终状态。

## Hook contract

- 同一事件的回调严格按注册顺序串行执行。
- 回调可以是同步函数或 `async def` 函数。
- 回调返回 `None` 表示不干预；返回其他值由触发方决定是否使用。
- `BEFORE_TOOL` 中任意回调返回 `HookDecision(allow=False)` 即拒绝执行；多个回调仍全部执行，便于审计同时完成。
- 回调抛出的普通异常包装为 `HookError`；工具前置钩子异常按失败关闭处理，不进工具。其他生命周期钩子异常终止当前任务。
- 注册表不提供隐式去重，重复注册表示确实需要执行两次。

## Data flow

```text
USER_PROMPT
  -> compress
  -> AFTER_COMPRESSION
  -> BEFORE_MODEL
  -> model stream
  -> AFTER_MODEL
  -> BEFORE_TOOL
       -> deny: 返回拒绝结果
       -> allow: execute -> truncate -> AFTER_TOOL
  -> next turn
  -> STOP
```

## Compatibility

- `HookRegistry` 是 `AgentLoop` 的可选构造参数，默认创建空注册表。
- 现有直接调用入口保持不变。
- 权限检查从 `_query_loop()` 的直接调用迁移到内置前置工具钩子，行为仍是原有四层权限审查。
- 现有测试不要求新增构造参数。

## Testing

- 注册顺序、同步/异步回调、异常包装和拒绝决定分别有单元测试。
- 主循环测试验证事件顺序、权限钩子拒绝后工具不执行、停止钩子只触发一次。
- 运行完整测试套件，确认权限、并发工具执行、子 Agent 和压缩路径不回归。

## Acceptance criteria

- 不注册任何自定义钩子时，现有行为不变。
- 新增一个观察型能力只需注册回调，无需修改 `_query_loop()`。
- 工具前置拒绝路径有测试，且不会绕过原有权限检查。
- 代码文档明确钩子执行顺序和异常语义。
