# 记忆闭环的端到端证据（真模型、跨进程）

> 结论一句话：**上一轮写进长期记忆的约定，下一轮（新进程、没有对话历史）被自动召回并
> 注入上下文，模型据此答对；关掉注入，同一个问题就答不出来。**
>
> 采集时间：2026-10-07 ｜ 模型：deepseek-chat ｜ 三轮总成本 **$0.0025**
> 工作目录是一个全新的空目录（只有 `.env`），所以记忆库是干净的、跨进程共享的。

## 第一轮：把约定写进长期记忆

```text
> 请把这条项目约定写进长期记忆：跑 pytest 之前必须先设置环境变量禁用联网下载。
  用 Remember 工具，memory_type 用 procedural。写完复述一遍。

[Remember] => Recorded [procedural] memory.
已写入长期记忆。复述一遍这条约定：…必须先设置环境变量以禁用联网下载…

完成: 2 turns, 4,519 tokens, $0.0013
        前缀缓存: 2,048 read / 0 write
```

落库确认（直接查 SQLite）：

```text
('procedural', '项目约定：运行 pytest 之前，必须先设置环境变量以禁用联网下载（避免…）', 1)
('episodic',   'Task: 请把这条项目约定写进长期记忆… | 2 turns | 4519 tokens …', 1)
```

## 第二轮：新进程、问同一件事

提示词里自动带上的记忆块（用 `build_system_prompt` 复现，不必调模型就能核对）：

```text
## 可能相关的历史记忆（仅供参考，若与当前任务不符请忽略）
- [事件] Task: 请把这条项目约定写进长期记忆…
- [经验] 项目约定：运行 pytest 之前，必须先设置环境变量以禁用联网下载…
```

### 对照实验（同一个问题、同一份记忆库）

| 条件 | 回答 | 结论 |
|---|---|---|
| **带注入** | 「按项目约定，跑 pytest 之前必须先设置环境变量以禁用联网下载（避免测试过程中自动下载依赖、模型或数据集）」 | 复述了**跨会话**记住的约定 ✅ |
| `--no-memory-inject` | 「项目约定要看 CLAUDE.md，但当前项目没有这个文件，所以我无法确认。需要我先创建它吗？」 | 答不出，反而反问 ❌ |

成本：带注入 1 turn / 2,268 tokens / $0.0006（前缀缓存 256 read）；对照组 1 turn / 2,122 tokens / $0.0006（1,920 read）。

## 这份证据顺带说明的两件事

1. **自动注入这一环是必要的**，不是锦上添花：对照组显示模型**没有**主动去调 `RecallMemory`，
   而是转去问 CLAUDE.md —— 也就是"存了但不会自己想起来查"。这正是当初补"读"这一端的理由，
   现在有实测。
2. **前缀缓存是可核验的**：每轮输出都带 `前缀缓存: N read / 0 write`，从服务端 usage 读回
   （`core/model_adapter.py` 的 `cached_tokens`）——简历上那句"命中量从服务端读回"由此坐实。

## 复现方式

```bash
mkdir /tmp/mem_e2e && cd /tmp/mem_e2e && cp <minicode>/.env .
python <minicode>/main.py "请把这条项目约定写进长期记忆：跑 pytest 之前必须先设置环境变量禁用联网下载。用 Remember 工具，memory_type 用 procedural。写完复述一遍。" --max-turns 6 --max-cost 0.05
python <minicode>/main.py "按项目约定，跑 pytest 之前要注意什么？一句话回答。" --max-turns 4 --max-cost 0.05
python <minicode>/main.py "按项目约定，跑 pytest 之前要注意什么？一句话回答。" --max-turns 4 --max-cost 0.05 --no-memory-inject
```
