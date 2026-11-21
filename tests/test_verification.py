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
async def test_verifier_reports_timeout():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "import time; time.sleep(2)"',
        timeout_seconds=0.1,
    )

    result = await verifier.verify()

    assert result.passed is False
    assert result.timed_out is True
