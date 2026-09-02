"""Token 成本基准测试 — 四级压缩的实测压缩比

测量 ContextCompressor 四级压缩各自能把 N token 压到多少。
用模拟长会话（读大文件 + 跑命令）构造 ~60K token，触发各级压缩。

不消耗真实 API — 纯本地压缩算法测量，可重复、零成本。

用法：
    python benchmarks/record.py
"""

import asyncio
import sys
from pathlib import Path

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 修复 Windows GBK 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.state import Message, LoopState
from capabilities.compression import (
    ContextCompressor, count_tokens, CompressionConfig,
)


def _simulate_file_content(lines: int = 40) -> str:
    """生成一段模拟文件内容（~40 行代码，约 1000 字符 / ~400 token）。"""
    template = [
        "def process_data(input_list, config=None):",
        "    \"\"\"Process input data through the pipeline.\"\"\"",
        "    results = []",
        "    for i, item in enumerate(input_list):",
        "        if config and config.get('skip_empty', True):",
        "            if not item or item.strip() == '':",
        "                continue",
        "        normalized = normalize(item)",
        "        transformed = transform(normalized, config)",
        "        validated = validate(transformed)",
        "        if validated:",
        "            results.append(transformed)",
        "        else:",
        "            log_warning(f'skipped item {i}: validation failed')",
        "    return results",
        "",
        "def normalize(value):",
        "    return str(value).lower().strip()",
        "",
        "def transform(value, config):",
        "    prefix = config.get('prefix', '') if config else ''",
        "    return f'{prefix}{value}'",
        "",
        "def validate(value):",
        "    return len(value) > 0 and value is not None",
        "",
        "def log_warning(msg):",
        "    print(f'[WARNING] {msg}')",
    ]
    # 重复填充到指定行数
    result = []
    while len(result) < lines:
        result.extend(template)
    return "\n".join(result[:lines])


def build_long_session(num_files: int = 30, lines_per_file: int = 40) -> LoopState:
    """构造一个模拟长会话：读 N 个大文件，每个返回约 400 token。

    30 个文件 × 400 token ≈ 12K token（tool 结果），加上 assistant 消息，
    总上下文超过压缩触发阈值，能验证各级压缩。
    """
    messages = [
        Message(role="system", content="You are MiniCode, an AI coding agent."),
        Message(role="user", content="Analyze all files in the project and report a summary."),
    ]

    for i in range(num_files):
        # 模拟 assistant 调用 Read 工具
        messages.append(Message(
            role="assistant",
            content=f"Reading file_{i}.py to understand its structure.",
            tool_calls=[{"id": f"call_{i}", "type": "function",
                         "function": {"name": "Read",
                                      "arguments": f'{{"file_path": "file_{i}.py"}}'}}],
        ))
        # 模拟工具返回文件内容
        content = _simulate_file_content(lines_per_file)
        messages.append(Message(
            role="tool", content=content, tool_call_id=f"call_{i}",
        ))

    return LoopState(messages=tuple(messages))


async def benchmark_compression():
    """测量四级压缩各自的压缩比。"""
    compressor = ContextCompressor(CompressionConfig(max_context_tokens=60_000))

    # 构造长会话（200 个文件，接近真实 60K 压缩阈值）
    state = build_long_session(num_files=200)
    tokens_before = count_tokens(state.messages)

    print("=" * 64)
    print("Token 压缩基准测试（模拟长会话：读 200 个文件）")
    print("=" * 64)
    print(f"{'级别':<22} {'消息数':>8} {'Token 数':>10} {'压缩比':>10}")
    print("-" * 64)
    print(f"{'原始（无压缩）':<22} {len(state.messages):>8} {tokens_before:>10,} {'—':>10}")

    # T2: Snip（工具输出占位符替换）
    snip_state = await compressor._snip(state)
    tokens_snip = count_tokens(snip_state.messages)
    print(f"{'Snip（占位替换）':<22} {len(snip_state.messages):>8} "
          f"{tokens_snip:>10,} {_ratio(tokens_before, tokens_snip):>10}")

    # T3: Collapse（掐头去尾 + 摘要）
    collapse_state = await compressor._collapse(state)
    tokens_collapse = count_tokens(collapse_state.messages)
    print(f"{'Collapse（掐头去尾）':<22} {len(collapse_state.messages):>8} "
          f"{tokens_collapse:>10,} {_ratio(tokens_before, tokens_collapse):>10}")

    # T4: Autocompact（全量压缩）
    compact_state = await compressor.force_autocompact(state)
    tokens_compact = count_tokens(compact_state.messages)
    print(f"{'Autocompact（全量压缩）':<22} {len(compact_state.messages):>8} "
          f"{tokens_compact:>10,} {_ratio(tokens_before, tokens_compact):>10}")

    print("-" * 64)
    print(f"\n结论：")
    print(f"  Snip 压缩比:      {_percent(tokens_before, tokens_snip)}")
    print(f"  Collapse 压缩比:  {_percent(tokens_before, tokens_collapse)}")
    print(f"  Autocompact 压缩比: {_percent(tokens_before, tokens_compact)}")
    print(f"\n  （Token 成本降低 = 压缩比，即长会话场景下减少的重复传输）")

    return {
        "before": tokens_before,
        "snip": tokens_snip,
        "collapse": tokens_collapse,
        "autocompact": tokens_compact,
    }


def _ratio(before: int, after: int) -> str:
    if before == 0:
        return "—"
    return f"{after/before:.1%}"


def _percent(before: int, after: int) -> str:
    if before == 0:
        return "—"
    return f"{(before - after) / before:.0%}"


if __name__ == "__main__":
    asyncio.run(benchmark_compression())
