"""Fixed coding tasks for real-agent evaluation.

Each task contains the initial repository, a hidden acceptance test, and a
reference solution used only to validate that the fixture is solvable. Hidden
tests are never materialized into the agent workspace by the runner.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingTask:
    id: str
    category: str
    prompt: str
    files: dict[str, str]
    hidden_test: str
    solution_files: dict[str, str]
    timeout_seconds: float = 45.0
    session_timeout_seconds: float = 180.0
    max_turns: int = 10
    max_cost_usd: float = 0.10


@dataclass(frozen=True)
class HiddenTestResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def materialize_task(task: CodingTask, root: Path, *, solution: bool = False) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = task.solution_files if solution else task.files
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def run_hidden_test(root: Path, hidden_test: str, timeout_seconds: float = 45.0):
    tests_dir = Path(root) / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_path = tests_dir / "test_hidden_acceptance.py"
    test_path.write_text(hidden_test, encoding="utf-8", newline="\n")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_hidden_acceptance.py"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return HiddenTestResult(
            exit_code=None,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "") + f"\nTimed out after {timeout_seconds}s",
            timed_out=True,
        )
    return HiddenTestResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def validate_task_fixture(task: CodingTask) -> tuple[bool, bool, str]:
    with tempfile.TemporaryDirectory(prefix=f"minicode-fixture-{task.id}-") as raw:
        root = Path(raw)
        materialize_task(task, root)
        baseline = run_hidden_test(root, task.hidden_test, task.timeout_seconds)

    with tempfile.TemporaryDirectory(prefix=f"minicode-solution-{task.id}-") as raw:
        root = Path(raw)
        materialize_task(task, root, solution=True)
        solution = run_hidden_test(root, task.hidden_test, task.timeout_seconds)

    detail = (
        f"baseline exit={baseline.exit_code} timed_out={baseline.timed_out}\n"
        f"{baseline.stdout}\n{baseline.stderr}\n"
        f"solution exit={solution.exit_code} timed_out={solution.timed_out}\n"
        f"{solution.stdout}\n{solution.stderr}"
    )
    return baseline.exit_code == 0, solution.exit_code == 0, detail


CODING_TASKS: tuple[CodingTask, ...] = (
    CodingTask(
        id="clamp_bounds",
        category="boundaries",
        prompt="修复 clamp(value, low, high)：保留区间两端，low > high 时抛出 ValueError。",
        files={"math_utils.py": "def clamp(value, low, high):\n    if low > high:\n        raise ValueError('low must be <= high')\n    return min(max(value, low), high - 1)\n"},
        solution_files={"math_utils.py": "def clamp(value, low, high):\n    if low > high:\n        raise ValueError('low must be <= high')\n    return min(max(value, low), high)\n"},
        hidden_test="""import pytest
from math_utils import clamp

def test_clamp_includes_upper_bound():
    assert clamp(-5, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
    assert clamp(15, 0, 10) == 10

def test_clamp_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        clamp(1, 2, 1)
""",
    ),
    CodingTask(
        id="paginate_last_item",
        category="boundaries",
        prompt="修复分页函数，确保每页恰好包含 size 条数据，且最后一页返回全部剩余元素。",
        files={"pagination.py": "def paginate(items, page, size):\n    if page < 1 or size < 1:\n        raise ValueError('page and size must be positive')\n    start = (page - 1) * size\n    end = page * size - 1\n    return list(items[start:end])\n"},
        solution_files={"pagination.py": "def paginate(items, page, size):\n    if page < 1 or size < 1:\n        raise ValueError('page and size must be positive')\n    start = (page - 1) * size\n    end = page * size\n    return list(items[start:end])\n"},
        hidden_test="""from pagination import paginate

def test_pages_include_full_slice():
    data = list(range(5))
    assert paginate(data, 1, 2) == [0, 1]
    assert paginate(data, 2, 2) == [2, 3]
    assert paginate(data, 3, 2) == [4]
""",
    ),
    CodingTask(
        id="parse_compound_duration",
        category="parsing",
        prompt="让 parse_duration 支持组合时长，例如 1h30m、2m15s；非法文本必须抛出 ValueError。",
        files={"duration.py": "import re\n\ndef parse_duration(text):\n    match = re.fullmatch(r'(\\d+)([hms])', text.strip())\n    if not match:\n        raise ValueError('invalid duration')\n    value, unit = match.groups()\n    return int(value) * {'s': 1, 'm': 60, 'h': 3600}[unit]\n"},
        solution_files={"duration.py": "import re\n\ndef parse_duration(text):\n    text = text.strip()\n    if not text:\n        raise ValueError('invalid duration')\n    total = 0\n    position = 0\n    for match in re.finditer(r'(\\d+)([hms])', text):\n        if match.start() != position:\n            raise ValueError('invalid duration')\n        value, unit = match.groups()\n        total += int(value) * {'s': 1, 'm': 60, 'h': 3600}[unit]\n        position = match.end()\n    if position != len(text):\n        raise ValueError('invalid duration')\n    return total\n"},
        hidden_test="""import pytest
from duration import parse_duration

def test_compound_units():
    assert parse_duration('1h30m') == 5400
    assert parse_duration('2m15s') == 135

def test_invalid_text_rejected():
    for value in ('', '10', '1h bad'):
        with pytest.raises(ValueError):
            parse_duration(value)
""",
    ),
    CodingTask(
        id="merge_intervals",
        category="algorithms",
        prompt="修复 merge_intervals：先排序，并合并重叠或首尾相接的区间。",
        files={"intervals.py": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    result = [tuple(intervals[0])]\n    for start, end in intervals[1:]:\n        last_start, last_end = result[-1]\n        if start < last_end:\n            result[-1] = (last_start, max(last_end, end))\n        else:\n            result.append((start, end))\n    return result\n"},
        solution_files={"intervals.py": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    ordered = sorted((int(start), int(end)) for start, end in intervals)\n    result = [ordered[0]]\n    for start, end in ordered[1:]:\n        last_start, last_end = result[-1]\n        if start <= last_end:\n            result[-1] = (last_start, max(last_end, end))\n        else:\n            result.append((start, end))\n    return result\n"},
        hidden_test="""from intervals import merge_intervals

def test_unsorted_and_touching_intervals_merge():
    assert merge_intervals([(5, 7), (1, 3), (3, 5), (10, 12)]) == [(1, 7), (10, 12)]
    assert merge_intervals([]) == []
""",
    ),
    CodingTask(
        id="parse_quoted_csv",
        category="parsing",
        prompt="修复 parse_csv_row，使带双引号且包含逗号的字段被正确解析。",
        files={"csv_row.py": "def parse_csv_row(line):\n    return [field.strip() for field in line.split(',')]\n"},
        solution_files={"csv_row.py": "import csv\nimport io\n\ndef parse_csv_row(line):\n    return next(csv.reader(io.StringIO(line)))\n"},
        hidden_test="""from csv_row import parse_csv_row

def test_quoted_comma_and_spaces():
    assert parse_csv_row('1,"Smith, Jane",admin') == ['1', 'Smith, Jane', 'admin']
    assert parse_csv_row('a,b,c') == ['a', 'b', 'c']
""",
    ),
    CodingTask(
        id="lru_get_recency",
        category="data_structures",
        prompt="修复 LRUCache.get：命中缓存时也必须刷新最近使用顺序。",
        files={"lru.py": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.data = OrderedDict()\n    def get(self, key):\n        return self.data.get(key)\n    def put(self, key, value):\n        self.data[key] = value\n        self.data.move_to_end(key)\n        if len(self.data) > self.capacity:\n            self.data.popitem(last=False)\n"},
        solution_files={"lru.py": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.data = OrderedDict()\n    def get(self, key):\n        if key not in self.data:\n            return None\n        value = self.data[key]\n        self.data.move_to_end(key)\n        return value\n    def put(self, key, value):\n        self.data[key] = value\n        self.data.move_to_end(key)\n        if len(self.data) > self.capacity:\n            self.data.popitem(last=False)\n"},
        hidden_test="""from lru import LRUCache

def test_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put('a', 1); cache.put('b', 2)
    assert cache.get('a') == 1
    cache.put('c', 3)
    assert cache.get('b') is None
    assert cache.get('a') == 1
    assert cache.get('c') == 3
""",
    ),
    CodingTask(
        id="exponential_retry_delay",
        category="reliability",
        prompt="修复 retry_delay，使第 1 次重试等待 base，之后按 2 倍增长并受 max_delay 限制。",
        files={"retry_delay.py": "def retry_delay(attempt, base, max_delay):\n    if attempt < 1:\n        raise ValueError('attempt must be positive')\n    return min(max_delay, base * attempt)\n"},
        solution_files={"retry_delay.py": "def retry_delay(attempt, base, max_delay):\n    if attempt < 1:\n        raise ValueError('attempt must be positive')\n    return min(max_delay, base * (2 ** (attempt - 1)))\n"},
        hidden_test="""from retry_delay import retry_delay

def test_exponential_backoff():
    assert [retry_delay(i, 100, 500) for i in range(1, 5)] == [100, 200, 400, 500]
""",
    ),
    CodingTask(
        id="greedy_word_wrap",
        category="algorithms",
        prompt="实现 wrap_text：按单词贪心换行，每行长度不超过 width；单词本身超长时单独保留。",
        files={"textwrap_user.py": "def wrap_text(text, width):\n    if width < 1:\n        raise ValueError('width must be positive')\n    return [text]\n"},
        solution_files={"textwrap_user.py": "def wrap_text(text, width):\n    if width < 1:\n        raise ValueError('width must be positive')\n    lines = []\n    current = ''\n    for word in text.split():\n        if not current:\n            current = word\n        elif len(current) + 1 + len(word) <= width:\n            current += ' ' + word\n        else:\n            lines.append(current)\n            current = word\n    if current:\n        lines.append(current)\n    return lines\n"},
        hidden_test="""from textwrap_user import wrap_text

def test_greedy_wrap():
    assert wrap_text('one two three four', 7) == ['one two', 'three', 'four']
    assert wrap_text('abcdefgh ij', 4) == ['abcdefgh', 'ij']
    assert wrap_text('', 8) == []
""",
    ),
    CodingTask(
        id="normalize_relative_path",
        category="parsing",
        prompt="修复 normalize_relative_path：去掉空段和 .，统一斜杠；任何 .. 段都视为越界并抛出 ValueError。",
        files={"path_text.py": "def normalize_relative_path(path):\n    return path.replace('\\\\', '/')\n"},
        solution_files={"path_text.py": "def normalize_relative_path(path):\n    parts = []\n    for part in path.replace('\\\\', '/').split('/'):\n        if part in ('', '.'):\n            continue\n        if part == '..':\n            raise ValueError('path escapes root')\n        parts.append(part)\n    return '/'.join(parts)\n"},
        hidden_test="""import pytest
from path_text import normalize_relative_path

def test_normalizes_common_segments():
    assert normalize_relative_path('./src//app.py') == 'src/app.py'
    assert normalize_relative_path('src\\\\lib\\\\x.py') == 'src/lib/x.py'

def test_rejects_parent_segment():
    with pytest.raises(ValueError):
        normalize_relative_path('src/../secret')
""",
    ),
    CodingTask(
        id="top_k_descending",
        category="algorithms",
        prompt="修复 top_k，返回最大的 k 个值并按降序排列。",
        files={"ranking.py": "def top_k(values, k):\n    if k < 0:\n        raise ValueError('k must be non-negative')\n    return sorted(values)[:k]\n"},
        solution_files={"ranking.py": "def top_k(values, k):\n    if k < 0:\n        raise ValueError('k must be non-negative')\n    return sorted(values, reverse=True)[:k]\n"},
        hidden_test="""from ranking import top_k

def test_top_k_is_descending():
    assert top_k([3, 1, 9, 7], 2) == [9, 7]
    assert top_k([], 3) == []
""",
    ),
    CodingTask(
        id="chunked_tail",
        category="boundaries",
        prompt="修复 chunked，保证序列末尾不足 size 的元素也会形成一个分块。",
        files={"chunking.py": "def chunked(seq, size):\n    if size < 1:\n        raise ValueError('size must be positive')\n    return [list(seq[i:i + size]) for i in range(0, len(seq) - 1, size)]\n"},
        solution_files={"chunking.py": "def chunked(seq, size):\n    if size < 1:\n        raise ValueError('size must be positive')\n    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]\n"},
        hidden_test="""from chunking import chunked

def test_tail_chunk_is_preserved():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunked([], 3) == []
""",
    ),
    CodingTask(
        id="binary_search_index",
        category="algorithms",
        prompt="修复 binary_search，返回目标索引，找不到返回 -1，不能在循环中停滞。",
        files={"binary_search.py": "def binary_search(values, target):\n    lo, hi = 0, len(values)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if values[mid] < target:\n            lo = mid\n        elif values[mid] > target:\n            hi = mid\n        else:\n            return mid\n    return -1\n"},
        solution_files={"binary_search.py": "def binary_search(values, target):\n    lo, hi = 0, len(values) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if values[mid] < target:\n            lo = mid + 1\n        elif values[mid] > target:\n            hi = mid - 1\n        else:\n            return mid\n    return -1\n"},
        timeout_seconds=5.0,
        hidden_test="""from binary_search import binary_search

def test_search_found_and_missing():
    values = [1, 3, 5, 7, 9]
    assert binary_search(values, 1) == 0
    assert binary_search(values, 9) == 4
    assert binary_search(values, 4) == -1
    assert binary_search([], 1) == -1
""",
    ),
    CodingTask(
        id="recursive_flatten",
        category="data_structures",
        prompt="实现 flatten，递归展开任意深度的列表，保留非列表元素顺序。",
        files={"flattening.py": "def flatten(values):\n    result = []\n    for item in values:\n        if isinstance(item, list):\n            result.extend(item)\n        else:\n            result.append(item)\n    return result\n"},
        solution_files={"flattening.py": "def flatten(values):\n    result = []\n    for item in values:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n"},
        hidden_test="""from flattening import flatten

def test_flatten_arbitrary_depth():
    assert flatten([1, [2, [3, [4]], 5], 6]) == [1, 2, 3, 4, 5, 6]
""",
    ),
    CodingTask(
        id="config_port_validation",
        category="multi_file",
        prompt="修复 build_config：host 必须非空，port 必须转换为 1..65535 的整数，否则抛出 ValueError。",
        files={"models.py": "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass AppConfig:\n    host: str\n    port: int\n", "app_config.py": "from models import AppConfig\n\ndef build_config(values):\n    return AppConfig(values['host'], values.get('port', 8000))\n"},
        solution_files={"models.py": "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass AppConfig:\n    host: str\n    port: int\n", "app_config.py": "from models import AppConfig\n\ndef build_config(values):\n    host = str(values.get('host', '')).strip()\n    if not host:\n        raise ValueError('host is required')\n    try:\n        port = int(values.get('port', 8000))\n    except (TypeError, ValueError) as exc:\n        raise ValueError('invalid port') from exc\n    if not 1 <= port <= 65535:\n        raise ValueError('invalid port')\n    return AppConfig(host, port)\n"},
        hidden_test="""import pytest
from app_config import build_config

def test_builds_valid_config():
    assert build_config({'host': ' api ', 'port': '443'}).port == 443

def test_rejects_invalid_values():
    for values in ({'host': ''}, {'host': 'api', 'port': 'bad'}, {'host': 'api', 'port': 70000}):
        with pytest.raises(ValueError):
            build_config(values)
""",
    ),
    CodingTask(
        id="event_bus_snapshot",
        category="state",
        prompt="修复 EventBus.publish：本轮发布使用订阅快照，发布过程中新增的监听器不能收到当前事件；取消订阅要继续生效。",
        files={"event_bus.py": "from collections import defaultdict\n\nclass EventBus:\n    def __init__(self):\n        self.listeners = defaultdict(list)\n    def subscribe(self, event, callback):\n        self.listeners[event].append(callback)\n    def unsubscribe(self, event, callback):\n        if callback in self.listeners[event]:\n            self.listeners[event].remove(callback)\n            return True\n        return False\n    def publish(self, event, payload):\n        for callback in self.listeners.get(event, []):\n            callback(payload)\n"},
        solution_files={"event_bus.py": "from collections import defaultdict\n\nclass EventBus:\n    def __init__(self):\n        self.listeners = defaultdict(list)\n    def subscribe(self, event, callback):\n        self.listeners[event].append(callback)\n    def unsubscribe(self, event, callback):\n        if callback in self.listeners[event]:\n            self.listeners[event].remove(callback)\n            return True\n        return False\n    def publish(self, event, payload):\n        for callback in tuple(self.listeners.get(event, [])):\n            callback(payload)\n"},
        hidden_test="""from event_bus import EventBus

def test_publish_uses_subscription_snapshot():
    bus = EventBus()
    seen = []
    def late(payload):
        seen.append(('late', payload))
    def first(payload):
        seen.append(('first', payload))
        bus.subscribe('x', late)
    bus.subscribe('x', first)
    bus.publish('x', 1)
    assert seen == [('first', 1)]
    bus.publish('x', 2)
    assert seen[-1] == ('late', 2)
""",
    ),
    CodingTask(
        id="retry_policy_multi_file",
        category="multi_file",
        prompt="修复多文件重试策略：retry_plan 应调用 policies.exponential_delay，生成正确的指数退避序列。",
        files={"policies.py": "def exponential_delay(attempt, base, max_delay):\n    return min(max_delay, base * (2 ** attempt))\n", "retry.py": "from policies import exponential_delay\n\ndef retry_plan(attempts, base=1, max_delay=10):\n    if attempts < 0:\n        raise ValueError('attempts must be non-negative')\n    return [exponential_delay(i, base, max_delay) for i in range(1, attempts + 1)]\n"},
        solution_files={"policies.py": "def exponential_delay(attempt, base, max_delay):\n    return min(max_delay, base * (2 ** (attempt - 1)))\n", "retry.py": "from policies import exponential_delay\n\ndef retry_plan(attempts, base=1, max_delay=10):\n    if attempts < 0:\n        raise ValueError('attempts must be non-negative')\n    return [exponential_delay(i, base, max_delay) for i in range(1, attempts + 1)]\n"},
        hidden_test="""from retry import retry_plan

def test_retry_plan_uses_shared_policy():
    assert retry_plan(4, base=2, max_delay=10) == [2, 4, 8, 10]
""",
    ),
    CodingTask(
        id="rate_limiter_boundary",
        category="state",
        prompt="修复 RateLimiter.allow：窗口边界上的旧请求应过期，limit 和 per_seconds 参数必须校验。",
        files={"rate_limit.py": "class RateLimiter:\n    def __init__(self, limit, per_seconds, clock):\n        self.limit = limit\n        self.per_seconds = per_seconds\n        self.clock = clock\n        self.events = []\n    def allow(self):\n        now = self.clock()\n        cutoff = now - self.per_seconds\n        self.events = [stamp for stamp in self.events if stamp >= cutoff]\n        if len(self.events) < self.limit:\n            self.events.append(now)\n            return True\n        return False\n"},
        solution_files={"rate_limit.py": "class RateLimiter:\n    def __init__(self, limit, per_seconds, clock):\n        if limit < 1 or per_seconds <= 0:\n            raise ValueError('invalid rate limit')\n        self.limit = limit\n        self.per_seconds = per_seconds\n        self.clock = clock\n        self.events = []\n    def allow(self):\n        now = self.clock()\n        cutoff = now - self.per_seconds\n        self.events = [stamp for stamp in self.events if stamp > cutoff]\n        if len(self.events) < self.limit:\n            self.events.append(now)\n            return True\n        return False\n"},
        hidden_test="""import pytest
from rate_limit import RateLimiter

def test_window_boundary_expires_old_request():
    now = [0]
    limiter = RateLimiter(1, 10, lambda: now[0])
    assert limiter.allow() is True
    assert limiter.allow() is False
    now[0] = 10
    assert limiter.allow() is True

def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0, 10, lambda: 0)
""",
    ),
    CodingTask(
        id="inventory_cancel_restock",
        category="multi_file",
        prompt="修复订单取消：cancel_order 必须把已预留库存同时从未预留和已预留数量中恢复。",
        files={"inventory.py": "def reserve(stock, sku, quantity):\n    if quantity < 1:\n        raise ValueError('quantity must be positive')\n    if stock['available'].get(sku, 0) < quantity:\n        raise ValueError('insufficient stock')\n    stock['available'][sku] -= quantity\n    stock['reserved'][sku] = stock['reserved'].get(sku, 0) + quantity\n\ndef release(stock, sku, quantity):\n    if stock['reserved'].get(sku, 0) < quantity:\n        raise ValueError('invalid reservation')\n    stock['reserved'][sku] -= quantity\n    stock['available'][sku] = stock['available'].get(sku, 0) + quantity\n", "orders.py": "def cancel_order(stock, sku, quantity):\n    stock['reserved'][sku] -= quantity\n"},
        solution_files={"inventory.py": "def reserve(stock, sku, quantity):\n    if quantity < 1:\n        raise ValueError('quantity must be positive')\n    if stock['available'].get(sku, 0) < quantity:\n        raise ValueError('insufficient stock')\n    stock['available'][sku] -= quantity\n    stock['reserved'][sku] = stock['reserved'].get(sku, 0) + quantity\n\ndef release(stock, sku, quantity):\n    if stock['reserved'].get(sku, 0) < quantity:\n        raise ValueError('invalid reservation')\n    stock['reserved'][sku] -= quantity\n    stock['available'][sku] = stock['available'].get(sku, 0) + quantity\n", "orders.py": "from inventory import release\n\ndef cancel_order(stock, sku, quantity):\n    release(stock, sku, quantity)\n"},
        hidden_test="""from inventory import reserve
from orders import cancel_order

def test_cancel_restores_available_stock():
    stock = {'available': {'sku': 5}, 'reserved': {}}
    reserve(stock, 'sku', 2)
    cancel_order(stock, 'sku', 2)
    assert stock == {'available': {'sku': 5}, 'reserved': {'sku': 0}}
""",
    ),
    CodingTask(
        id="dependency_cycle_detection",
        category="algorithms",
        prompt="修复 topo_sort：输出依赖优先的顺序，并在存在循环依赖时抛出 ValueError。",
        files={"topo.py": "def topo_sort(graph):\n    result = []\n    def visit(node):\n        for dependency in graph.get(node, []):\n            visit(dependency)\n        result.append(node)\n    for node in graph:\n        visit(node)\n    return result\n"},
        solution_files={"topo.py": "def topo_sort(graph):\n    result = []\n    state = {}\n    def visit(node):\n        marker = state.get(node, 0)\n        if marker == 1:\n            raise ValueError('cycle detected')\n        if marker == 2:\n            return\n        state[node] = 1\n        for dependency in graph.get(node, []):\n            visit(dependency)\n        state[node] = 2\n        result.append(node)\n    for node in graph:\n        visit(node)\n    return result\n"},
        hidden_test="""import pytest
from topo import topo_sort

def test_dependencies_come_first_and_cycle_rejected():
    result = topo_sort({'app': ['db'], 'db': ['core'], 'core': []})
    assert result.index('core') < result.index('db') < result.index('app')
    with pytest.raises(ValueError):
        topo_sort({'a': ['b'], 'b': ['a']})
""",
    ),
    CodingTask(
        id="safe_path_resolution",
        category="safety",
        prompt="修复 resolve_under：允许正常相对路径，但必须拒绝 ..、绝对路径和最终解析到根目录之外的结果。",
        files={"safe_paths.py": "from pathlib import Path\n\ndef resolve_under(root, user_path):\n    return (Path(root) / user_path).resolve()\n"},
        solution_files={"safe_paths.py": "from pathlib import Path\n\ndef resolve_under(root, user_path):\n    root_path = Path(root).resolve()\n    candidate = Path(user_path)\n    if candidate.is_absolute():\n        raise ValueError('absolute path is not allowed')\n    resolved = (root_path / candidate).resolve()\n    if not resolved.is_relative_to(root_path):\n        raise ValueError('path escapes root')\n    return resolved\n"},
        hidden_test="""import pytest
from safe_paths import resolve_under

def test_safe_paths_allowed(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    assert resolve_under(root, 'src/app.py') == (root / 'src/app.py').resolve()

def test_escape_rejected(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    for value in ('../secret', str(tmp_path / 'outside')):
        with pytest.raises(ValueError):
            resolve_under(root, value)
""",
    ),
)
