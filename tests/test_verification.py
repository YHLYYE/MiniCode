import asyncio
import sys

import pytest

from capabilities.verification import CommandVerifier


@pytest.mark.asyncio
async def test_verifier_passes_on_zero_exit():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "print(\'ok\')"',
        timeout_seconds=5,
    )

    result = await verifier.verify()

    assert result.passed is True
    assert result.exit_code == 0
    assert "ok" in result.output


@pytest.mark.asyncio
async def test_verifier_returns_tail_of_failed_output():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "print(123456789); raise SystemExit(2)"',
        timeout_seconds=5,
        max_output_chars=20,
    )

    result = await verifier.verify()

    assert result.passed is False
    assert result.exit_code == 2
    assert len(result.output) <= 20


@pytest.mark.asyncio
async def test_verifier_decodes_locale_output(monkeypatch):
    monkeypatch.setattr(
        "capabilities.verification.locale.getpreferredencoding",
        lambda _do_setlocale=False: "gbk",
    )
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "import sys; sys.stdout.buffer.write(\'中文\'.encode(\'gbk\'))"',
        timeout_seconds=5,
    )

    result = await verifier.verify()

    assert result.passed is True
    assert "中文" in result.output


@pytest.mark.asyncio
async def test_verifier_reports_timeout():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "import time; time.sleep(2)"',
        timeout_seconds=0.1,
    )

    result = await verifier.verify()

    assert result.passed is False
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_verifier_timeout_kills_descendant_process(tmp_path):
    marker = tmp_path / "late.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(1.5)\n"
        f"Path({str(marker)!r}).write_text('late')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        f"subprocess.run([sys.executable, {str(child)!r}])\n",
        encoding="utf-8",
    )
    verifier = CommandVerifier(
        f'"{sys.executable}" "{parent}"',
        timeout_seconds=0.2,
    )

    result = await verifier.verify()
    await asyncio.sleep(1.8)

    assert result.timed_out is True
    assert not marker.exists()
