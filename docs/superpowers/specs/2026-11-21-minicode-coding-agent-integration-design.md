# MiniCode 与编码 Agent 项目合并设计

日期：2026-11-21  
状态：待用户确认  
目标版本：MiniCode 外部验收与 Worktree 隔离 v1

## 1. 背景

MiniCode 当前已有 Agent Loop、上下文压缩、长期记忆、Skill 路由、权限和多 Agent 协作，但还有两个明确缺口：

1. 模型停止调用工具并输出文本时，引擎直接认为任务完成，缺少外部验收；
2. `general` 和 `team` 的写操作子 Agent 只隔离消息，不隔离文件，多个子任务可能修改同一工作区。

编码 Agent 项目中值得借鉴的是独立完成评估、任务级 Worktree、可恢复任务状态和相关测试；基础工具、记忆、Skill 和上下文压缩与 MiniCode 重合，不整体迁移。

当前 MiniCode 基线为 251 个测试通过、1 个跳过。

## 2. 非目标

- 不合并编码 Agent 项目的完整 17 章实现；
- 不引入 MCP、Cron、后台任务或 Agent Team 通讯协议；
- 不重写 MiniCode Agent Loop；
- 不自动把子 Agent 的 Worktree 合并到主分支；
- 不增加独立 LLM 评估器作为首版验收依据；
- 不改变未启用新功能时的现有行为。

## 3. 方案选择

### 方案一：只增加独立验收

范围小，能解决假完成，但写操作子 Agent 仍共享文件系统。不足以解决 MiniCode 当前的多 Agent 文件冲突问题。

### 方案二：外部验收 + Worktree 隔离

同时补齐完成判定和写隔离。两个能力均以可选参数启用，未启用时保持现有行为。采用此方案。

### 方案三：迁移完整 Harness 能力

把任务图、Workflow、Cron、MCP 和全部渐进章节一起合并。范围过大，会破坏 MiniCode 的现有架构和交付节奏。不采用。

## 4. 总体架构

新增两个独立能力模块：

```text
capabilities/
├── verification.py        外部验收命令、结果与重试
└── worktree.py            安全 Worktree 生命周期

core/state.py              新增验收重试与终止原因
core/agent_loop.py         在自然结束时调用可选 Verifier
capabilities/multi_agent.py  为写操作子 Agent 注入隔离工作区
main.py                    CLI 接入两个可选参数
```

能力之间保持解耦：

- Verifier 不需要知道 Worktree；
- Worktree 管理器不负责判断任务是否完成；
- AgentTool 只消耗 Worktree 能力；
- CLI 负责把用户提供的验收命令和隔离开关装配到运行时。

## 5. 外部验收

### 5.1 CLI

新增参数：

```text
--verify-command TEXT
--verify-timeout INT      默认 120 秒
--verify-retries INT      默认 2 次
```

当 `--verify-command` 未提供时：

- AgentLoop 不创建 Verifier；
- 保持现有“无工具调用并返回文本即完成”的行为；
- 原有测试不得改变。

验收命令只能由启动 MiniCode 的用户通过 CLI 提供，不能由模型生成或修改。

### 5.2 触发时机

只有模型本轮满足以下条件时才执行验收：

- 没有工具调用；
- assistant 有文本输出；
- 配置了 Verifier；
- 尚未达到验收重试上限。

验收不会在每次工具调用后自动重复。

### 5.3 验收执行

`CommandVerifier` 使用异步子进程执行命令：

- 在当前项目目录运行；
- 捕获 stdout、stderr 和退出码；
- 超时后终止进程并返回 `timed_out=true`；
- 输出保留尾部，设置固定字符上限，防止回灌爆上下文；
- 退出码为 0 才算通过；
- 异常转换为结构化失败结果，而不是直接中断会话。

### 5.4 失败回灌

验收失败时，向消息尾部加入一条 user 消息：

```text
[External verification failed]
command: ...
exit_code: ...
timed_out: ...
evidence:
...
请修复未通过的验收，不要只重复声称完成。
```

随后继续 Agent Loop。新增：

```text
ContinueReason.VERIFICATION_RETRY
TerminationReason.COMPLETED
TerminationReason.VERIFY_FAILED
TerminationReason.MAX_TURNS
TerminationReason.EMPTY_RESPONSE
TerminationReason.MAX_OUTPUT_TOKENS
```

现有 `LoopState` 增加只读 `termination_reason` 字段。DoneEvent 的终止语义必须可区分“自然完成”和“验收耗尽”。

### 5.5 重试上限

`--verify-retries` 表示验收失败后最多追加的修复轮数。

- 默认 2；
- 达到上限后不再继续循环；
- 返回 `VERIFY_FAILED`，保留最后一次退出码和证据；
- 不把验收失败伪装成成功完成。

## 6. Worktree 隔离

### 6.1 CLI

新增参数：

```text
--isolate-subagents
```

默认关闭，以保持现有行为。

启用后：

- `explore` 继续在主工作区只读执行；
- `general` 在独立 Worktree 中执行；
- `team` 的 research 在主工作区只读；
- `team` 的 coding 与 testing 共用同一个 Worktree；
- 主 Agent 仍在原工作区。

隔离策略由用户配置决定，不由模型自由选择。

### 6.2 启用条件

Worktree 管理必须满足：

- 当前目录位于 Git 仓库；
- `git status --porcelain` 为空；
- Worktree 名称经过安全校验；
- 目标目录位于 `.minicode/worktrees/`；
- 分支不存在或可以安全创建。

任一条件不满足时，写操作子 Agent 返回明确错误，不静默回退到主工作区。

如果仓库存在未提交改动，不能从 HEAD 创建工作树并把该工作树当成完整用户现场，因此本次拒绝隔离任务。

### 6.3 Worktree 生命周期

每次隔离任务包含：

- `task_id`：稳定哈希生成的短 ID；
- `branch`：`minicode/<slug>-<task_id>`；
- `path`：`.minicode/worktrees/<slug>-<task_id>`；
- `base_commit`：创建时的 HEAD；
- `status`：active、completed、failed、kept。

成功和失败都默认保留 Worktree，防止丢失子 Agent 现场。清理必须通过显式操作。

`.minicode/` 加入 `.gitignore`。

### 6.4 工具作用域

不使用进程级 `os.chdir`。每个隔离子 Agent 使用绑定工作区的工具实例：

- Read/Write/Edit：相对路径从 Worktree 根解析；绝对路径必须在 Worktree 内；
- Grep/Glob：默认根目录为 Worktree；
- Bash：cwd 固定为 Worktree，并拒绝明显的 `cd`、`..`、Windows 盘符和 POSIX 绝对路径；
- 拒绝 `../`、符号链接逃逸和跨 Worktree 路径。

Bash 的上述限制是 cwd + 词法防护，不是操作系统级沙箱。命令仍可能通过解释器或间接方式越过词法规则；因此本项目不承诺 Bash 的强隔离，不可信命令必须使用容器或其他 OS 级沙箱。

Project Memory、Skill 与模型适配器继续共享父级配置；文件、Shell 和搜索工具必须隔离。

### 6.5 提交与结果

隔离子 Agent 完成后：

- 若存在改动，提交到自己的分支；
- 返回 branch、commit、changed files、diff stat 和子 Agent 摘要；
- 不自动 merge、rebase 或 cherry-pick；
- 主 Agent 只收到以上结果，不接收完整子会话。

用户后续可以检查提交并手动应用到主工作区。

## 7. Team 流程

启用 Worktree 后：

```text
research  -> 主工作区，只读
coding    -> Worktree，写代码并提交
testing   -> 同一 Worktree，运行测试
```

如果 coding 失败：

- testing 不启动；
- Worktree 保留；
- 返回失败摘要和现场路径。

如果 testing 失败：

- Worktree 保留；
- 返回测试输出尾部；
- 不把分支报告为可安全应用。

如果 testing 通过：

- 返回绿色状态；
- 仍不自动合并。

## 8. 错误与恢复

- 验收命令超时：记录超时，继续把证据回灌模型；
- 验收命令不存在：结构化失败，不抛出到 CLI；
- Worktree 创建失败：子 Agent 不执行；
- Git 提交失败：保留工作树并返回错误；
- Worktree 清理失败：只报告错误，不删除主仓库；
- 子 Agent 取消：保留现场，状态标记 failed。

## 9. 测试设计

### 9.1 外部验收

- 验收通过后结束；
- 验收失败后继续一轮；
- 修复后第二次验收通过；
- 重试耗尽返回 `VERIFY_FAILED`；
- 命令超时返回结构化失败；
- 输出截断不会破坏消息格式；
- 未配置 Verifier 时行为与基线一致。

### 9.2 Worktree

- 非 Git 目录拒绝创建；
- dirty 工作区拒绝创建；
- 非法名称、绝对路径和 `../` 被拒绝；
- 正常创建与提交；
- `.minicode/worktrees` 被忽略；
- 失败保留现场；
- 不自动合并主分支。

### 9.3 AgentTool

- 默认模式行为不变；
- explore 永不创建 Worktree；
- general 启用隔离后写入独立目录；
- team 的 coding/testing 共用 Worktree；
- research 不写文件；
- 关闭隔离参数时不创建 Worktree；
- 子 Agent 只把摘要、branch、commit 和 diff stat回传父 Agent。

### 9.4 回归

- 原有 251 个通过测试继续通过；
- 原有 1 个跳过项状态不变；
- `python -m pytest -q` 全绿；
- Compileall 通过；
- README 和 DESIGN 只在功能实际验证后更新。

## 10. 验收标准

1. 未使用新参数时，MiniCode 行为与当前一致；
2. 配置验收命令后，模型无法在验收失败时结束任务；
3. 验收重试耗尽时返回明确失败原因；
4. 启用隔离后，文件与搜索工具不能访问主工作区；Bash 拦截常见逃逸语法，但明确标记为非沙箱；
5. Team 的 coding/testing 在同一隔离分支中工作；
6. Worktree 失败时不会静默退回主工作区；
7. 子 Agent 不自动合并；
8. 所有新增测试和原测试通过；
9. 文档明确这是可选能力，不声称自动合并或生产部署。

## 11. 真实性边界

允许的表述：

- MiniCode 新增可选的外部验收门；
- 可写子 Agent 可运行在独立 Git Worktree；
- 子 Agent 结果包含分支、提交和 diff，不自动合并；
- 两项能力有自动化和 Git 场景测试。

禁止的表述：

- 所有任务都默认隔离；
- 子 Agent 修改已经自动进入主分支；
- 验收器理解任意自然语言完成条件；
- 编码 Agent 项目的能力已全部迁移。
