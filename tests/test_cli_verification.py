from click.testing import CliRunner

from main import main


def test_help_lists_verification_and_isolation_flags():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--verify-command" in result.output
    assert "--verify-timeout" in result.output
    assert "--verify-retries" in result.output
    assert "--isolate-subagents" in result.output
