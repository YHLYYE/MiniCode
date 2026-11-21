# MiniCode 记忆作用域与结果反馈设计

日期：2026-11-21
状态：用户已确认设计
目标版本：MiniCode 记忆治理第一阶段

## 1. 背景

MiniCode 当前已有程序性、情景、用户画像三类长期记忆，使用 SQLite 和 BM25 检索，并记录检索次数与 `adoption_count`。现有实现还有两个问题：

1. 原始工具错误和普通任务过程日志也会进入跨会话候选，可能污染后续任务；
2. `adoption_count` 依据“任务结束时没有工具错误”更新，不能代表记忆确实帮助任务完成，更不能代表外部验收通过。

本阶段只修正作用域和结果反馈，不引入 Markdown 双写、模型重排或自动知识合并。

## 2. 目标

1. 每条记忆具有 `persistent` 或 `current_task` 作用域；
2. 自动提示词注入只使用持久记忆；
3. 临时记忆仍可由模型通过 `RecallMemory` 手动查询；
4. 注入后的任务结果按最终终止原因更新成功/失败计数；
5. 旧 SQLite 数据自动迁移，不丢失现有记忆；
6. 保持现有 BM25、严格证据门槛、类型名次融合、Top-3 和 600 字符限制。

## 3. 非目标

- 不迁移到 Markdown 主存储；
- 不增加 Markdown 双写；
- 不增加模型辅助重排；
- 不自动合并语义重复或过期记忆；
- 不新增向量数据库；
- 不改写 CLAUDE.md；
- 不改变 `TodoWrite` 或任务图存储。

## 4. 数据模型

`MemoryEntry` 新增：

```text
scope: "persistent" | "current_task" = "persistent"
success_count: int = 0
failure_count: int = 0
```

保留：

```text
retrieval_count
adoption_count
```

`adoption_count` 为兼容字段，不再作为主要结果指标。新排序优先使用：

```text
success_count - failure_count
```

SQLite 的 `memories` 表迁移新增：

```sql
ALTER TABLE memories ADD COLUMN scope TEXT DEFAULT 'persistent';
ALTER TABLE memories ADD COLUMN success_count INTEGER DEFAULT 0;
ALTER TABLE memories ADD COLUMN failure_count INTEGER DEFAULT 0;
```

迁移必须幂等。旧记录默认 `scope=persistent`，两个新计数从 0 开始。

## 5. 写入作用域

| 记忆来源 | 作用域 | 原因 |
|---|---|---|
| 用户纠正 | `persistent` | 长期偏好或规则 |
| 错误后成功修复形成的程序性经验 | `persistent` | 可复用修法 |
| 显式 `Remember` | `persistent` | 用户或模型明确要求保存 |
| 原始工具报错日志 | `current_task` | 只用于当前任务诊断 |
| 普通文件操作事件 | `current_task` | 过程日志不应跨任务注入 |
| 任务完成摘要 | `current_task` | 保留现场信息，但不能自动污染后续任务 |

`MemoryManager.record_*` 接受 `scope` 参数，默认 `persistent`。调用方为临时日志显式传入 `current_task`。

## 6. 召回与注入

`MemoryStore.search()` 增加 `scope: str | None` 过滤参数：

- `scope=None`：不限制，手动检索使用；
- `scope="persistent"`：只返回持久记忆，自动提示词注入使用。

主 Agent 的 `system_prompt_for(task)` 使用：

```python
memory_manager.search_sync(
    task,
    top_k=3,
    scope="persistent",
    min_similarity=0.10,
)
```

`RecallMemory` 默认不限制作用域，因此当前任务的错误日志仍可手动查询。

同一个任务只进行一次自动记忆注入。当前系统在每次 `AgentLoop.run()` 开始时构建一次 System Prompt，因此无需额外的全局会话去重状态。

## 7. 结果反馈

`_record_task_done` 增加显式 `success: bool` 参数，不再根据 `_task_errors` 推断记忆是否有效。

| TerminationReason | success |
|---|---|
| `COMPLETED` | `True` |
| `VERIFY_FAILED` | `False` |
| `MAX_TURNS` | `False` |
| `EMPTY_RESPONSE` | `False` |
| `MAX_OUTPUT_TOKENS` | `False` |

`MemoryManager.record_task_outcome(success)` 对上一次自动注入的记忆集合：

- 成功时 `success_count += 1`；
- 失败时 `failure_count += 1`；
- 空集合不做任何更新；
- 同一任务只结算一次。

工具报错后修复、最终正常完成或通过外部验收，仍记成功；验收失败或终止失败，记失败。

## 8. 排序

BM25、别名扩展、严格命中证据和跨类型名次融合保持不变。

条目内部排序加入温和的结果反馈：

```text
feedback = 1 + 0.05 * min(success_count, 5)
             - 0.03 * min(failure_count, 5)
feedback = max(0.5, feedback)
```

`adoption_count` 不再参与主要排序，避免旧语义继续影响结果。现有检索效果门槛不得下降。

## 9. 错误与兼容

- 旧数据库缺少新列：启动时幂等迁移；
- 未知 scope：写入拒绝，读取视为 `persistent` 或明确报错由实现选择固定规则；本设计规定读取时归一化为 `persistent`；
- 旧 JSON 记忆缺少新字段：补默认值；
- 结果计数更新失败：不能让任务执行崩溃；记录错误并保留任务结果；
- 损坏数据库：保持现有行为，不静默删除记忆。

## 10. 测试设计

1. SQLite 旧表迁移后保留原记录；
2. 新记录写入和读取 scope；
3. `current_task` 不进入严格自动注入；
4. `current_task` 可以通过手动检索获得；
5. `persistent` 正常进入自动注入；
6. 成功终止增加 success_count；
7. 验收失败增加 failure_count；
8. 工具报错后正常完成仍记成功；
9. 无注入记忆时不更新计数；
10. 旧 adoption_count 数据仍能读取；
11. 现有 BM25 召回门槛不下降；
12. 现有无关任务零误注入测试继续通过。

## 11. 验收标准

1. 自动注入只包含持久记忆；
2. 临时记忆可手动召回；
3. 成功/失败计数与最终终止原因一致；
4. 旧 SQLite 和 JSON 数据可继续使用；
5. BM25、Top-3、600 字符和严格证据门槛不变；
6. 全量测试、文档数字守护、编译和静态检查通过；
7. 文档不声称已实现语义重排、知识合并或记忆自进化。

## 12. 第二阶段候选

以下内容明确留到后续，不与本阶段混合：

- 离线合并重复和过期记忆；
- Markdown 审计导出；
- BM25 歧义时的条件式模型重排；
- 记忆版本、人工回滚界面。
