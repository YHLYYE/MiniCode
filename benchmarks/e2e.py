"""端到端 Token 成本对比 — 开压缩 vs 关压缩

模拟 Agent Loop 读 N 个文件的多轮会话，累计每轮 API 请求的 input token，
对比「开压缩」和「关压缩」的总 token 消耗。

诚实说明：这是「模拟端到端」（用 mock 模型行为 + 精确 token 计数），
不是真实 API 调用 —— 真实 API 跑 30 文件长任务需要几毛钱 + 几分钟，
而本脚本零成本、可重复，且能精确控制变量得出干净的对比。

用法：
    python benchmarks/e2e.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.state import Message, LoopState
from capabilities.compression import (
    ContextCompressor, count_tokens, CompressionConfig,
)

# 每个文件的内容 token 数（模拟读一个 ~2000 token 的文件）
FILE_TOKENS = 2000


def _make_file_content() -> str:
    """生成约 2000 token 的模拟文件内容（一段重复的代码）。"""
    block = """def process_batch(items, config=None, logger=None):
    \"\"\"Process a batch of items through the validation pipeline.\"\"\"
    results = []
    for idx, item in enumerate(items):
        if config and config.get('skip_empty', True):
            if item is None or str(item).strip() == '':
                continue
        normalized = normalize_record(item)
        validated = validate_record(normalized, config)
        if validated:
            results.append(normalized)
        elif logger:
            logger.warning(f'Item {idx} failed validation: {normalized}')
    return results

"""
    # 重复到约 2000 token（一个 block 约 90 token）
    return block * 22


async def simulate_session(num_files: int, max_context_tokens: int) -> dict:
    """模拟读 N 个文件的会话，返回累计 token 统计。

    每轮：assistant 调 Read → tool 返回文件内容 → 检查压缩。
    累计 input token = 每轮发送给 API 的上下文 token 总数。
    """
    compressor = ContextCompressor(
        CompressionConfig(max_context_tokens=max_context_tokens)
    )
    file_content = _make_file_content()

    state = LoopState(messages=(
        Message(role="system", content="You are MiniCode, an AI coding agent."),
        Message(role="user", content="Read and summarize every file in the project."),
    ))

    total_input_tokens = 0
    compression_events = []

    for i in range(num_files):
        # assistant 调 Read 工具
        state = state.add_message(Message(
            role="assistant",
            content=f"Reading file_{i}.py",
            tool_calls=[{"id": f"c{i}", "type": "function",
                         "function": {"name": "Read",
                                      "arguments": f'{{"file_path": "file_{i}.py"}}'}}],
        ))
        # tool 返回文件内容
        state = state.add_message(Message(
            role="tool", content=file_content, tool_call_id=f"c{i}",
        ))

        # 累计这轮的 input token（发送给 API 的上下文大小）
        total_input_tokens += count_tokens(state.messages)

        # 压缩检查
        before = count_tokens(state.messages)
        new_state = await compressor.compress_if_needed(state)
        after = count_tokens(new_state.messages)
        if after < before:
            compression_events.append((i + 1, before, after))
        state = new_state

    return {
        "total_input_tokens": total_input_tokens,
        "final_context_tokens": count_tokens(state.messages),
        "final_messages": len(state.messages),
        "compression_events": compression_events,
    }


async def main():
    NUM_FILES = 30
    COMPRESS_THRESHOLD = 20_000  # 开压缩：~10 个文件后触发
    NO_COMPRESS_THRESHOLD = 1_000_000  # 关压缩：永不触发

    print("=" * 66)
    print("端到端 Token 成本对比（模拟读 30 个文件，每个 ~2000 token）")
    print("=" * 66)

    # 关压缩
    print("\n[1/2] 关压缩（阈值 1M，永不触发）...")
    no_comp = await simulate_session(NUM_FILES, NO_COMPRESS_THRESHOLD)

    # 开压缩
    print("[2/2] 开压缩（阈值 20K，~10 文件后触发）...")
    with_comp = await simulate_session(NUM_FILES, COMPRESS_THRESHOLD)

    # 输出结果
    print("\n" + "=" * 66)
    print("结果对比")
    print("=" * 66)
    print(f"{'指标':<24} {'关压缩':>14} {'开压缩':>14}")
    print("-" * 66)
    print(f"{'累计 input token':<24} "
          f"{no_comp['total_input_tokens']:>14,} "
          f"{with_comp['total_input_tokens']:>14,}")
    print(f"{'最终上下文 token':<24} "
          f"{no_comp['final_context_tokens']:>14,} "
          f"{with_comp['final_context_tokens']:>14,}")
    print(f"{'最终消息数':<24} "
          f"{no_comp['final_messages']:>14} "
          f"{with_comp['final_messages']:>14}")

    saved = no_comp["total_input_tokens"] - with_comp["total_input_tokens"]
    pct = saved / no_comp["total_input_tokens"] * 100

    print("-" * 66)
    print(f"\n✅ Token 成本降低：{saved:,} tokens（{pct:.1f}%）")
    print(f"\n压缩触发记录（开压缩）：")
    if with_comp["compression_events"]:
        for round_num, before, after in with_comp["compression_events"]:
            print(f"  第 {round_num} 轮：{before:,} → {after:,} token "
                  f"（压掉 {(before-after)/before:.0%}）")
    else:
        print("  （未触发压缩）")


if __name__ == "__main__":
    asyncio.run(main())
