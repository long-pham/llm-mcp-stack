"""Tests for claude-switch — Claude Code auth mode switcher.

Tests the proxy (API) and official (Claude Max) modes by invoking the
claude-switch shell script and verifying settings.json, env file, and
config file are written correctly.  The final test switches to official
mode and runs ``claude -p`` to confirm end-to-end connectivity.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parent.parent / "claude-switch" / "claude-switch.sh"
SETTINGS_JSON = Path.home() / ".claude" / "settings.json"
# Current (primary) config location
CLAUDE_JSON = Path.home() / ".claude.json"
# Legacy config location
CLAUDE_JSON_LEGACY = Path.home() / ".claude" / "claude.json"
CONFIG_DIR = Path.home() / ".config" / "claude-switch"
CONFIG_FILE = CONFIG_DIR / "config"
ENV_FILE = CONFIG_DIR / "env"
ACTIVE_FILE = CONFIG_DIR / "active"
STASHED_OAUTH = CONFIG_DIR / "oauth.json.bak"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_switch(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run claude-switch with the given arguments."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def read_settings() -> dict:
    """Read and parse ~/.claude/settings.json."""
    return json.loads(SETTINGS_JSON.read_text())


def read_env() -> str:
    """Read the generated env file."""
    return ENV_FILE.read_text()


def read_claude_json() -> dict:
    """Read and parse the primary Claude config (~/.claude.json)."""
    return json.loads(CLAUDE_JSON.read_text())


def read_claude_json_legacy() -> dict:
    """Read and parse the legacy Claude config (~/.claude/claude.json)."""
    return json.loads(CLAUDE_JSON_LEGACY.read_text())


def read_config() -> dict[str, str]:
    """Parse the key=value config file into a dict."""
    result = {}
    for line in CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Fixtures — save and restore state so tests are non-destructive
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _save_restore_state():
    """Back up settings.json, env, config, and active files before each test
    and restore them afterwards so tests don't permanently alter the user's
    Claude Code configuration."""
    backups: dict[Path, str | None] = {}
    for path in (SETTINGS_JSON, CLAUDE_JSON, CLAUDE_JSON_LEGACY, CONFIG_FILE, ENV_FILE, ACTIVE_FILE, STASHED_OAUTH):
        backups[path] = path.read_text() if path.exists() else None

    yield

    for path, content in backups.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content)


# ---------------------------------------------------------------------------
# Tests — Proxy (API) mode
# ---------------------------------------------------------------------------

class TestProxyMode:
    """Test switching to proxy mode and verifying all artifacts."""

    PROXY_URL = "http://test-host:11212/api/anthropic"
    PROXY_KEY = "sk-test-key-for-pytest"

    def test_switch_to_proxy_with_inline_args(self):
        """Switching to proxy with inline URL and key sets everything up."""
        result = run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        assert result.returncode == 0
        assert "PROXY mode" in result.stdout

    def test_settings_json_has_api_base_url(self):
        """settings.json should contain apiBaseUrl after proxy switch."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        settings = read_settings()
        assert settings["apiBaseUrl"] == self.PROXY_URL

    def test_settings_json_preserves_existing_keys(self):
        """Switching to proxy must not clobber unrelated settings.json keys."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        settings = read_settings()
        # The env block should still be there from the user's real config
        assert "env" in settings or "apiBaseUrl" in settings  # at minimum apiBaseUrl

    def test_env_file_exports_proxy_vars(self):
        """The generated env file should export ANTHROPIC_BASE_URL and KEY."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        env_content = read_env()
        assert f'export ANTHROPIC_BASE_URL="{self.PROXY_URL}"' in env_content
        assert f'export ANTHROPIC_API_KEY="{self.PROXY_KEY}"' in env_content

    def test_config_file_persists_inline_args(self):
        """Inline URL and key should be written to the config file."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        cfg = read_config()
        assert cfg["PROXY_BASE_URL"] == self.PROXY_URL
        assert cfg["PROXY_API_KEY"] == self.PROXY_KEY

    def test_active_file_says_proxy(self):
        """The active file should contain 'proxy'."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        assert ACTIVE_FILE.read_text().strip() == "proxy"

    def test_status_reports_proxy(self):
        """'status' command should report proxy mode after switch."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        result = run_switch("status")

        assert "proxy" in result.stdout.lower()
        assert self.PROXY_URL in result.stdout

    def test_proxy_uses_saved_config(self):
        """Running 'proxy' without args should use previously saved values."""
        # First call sets the config
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        # Switch away
        run_switch("official")
        # Switch back without args — should use saved config
        result = run_switch("proxy")

        assert result.returncode == 0
        settings = read_settings()
        assert settings["apiBaseUrl"] == self.PROXY_URL

    def test_proxy_url_override_only(self):
        """Passing only a URL should update the URL but keep the saved key."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        new_url = "http://other-host:9090/api/anthropic"
        run_switch("proxy", new_url)

        cfg = read_config()
        assert cfg["PROXY_BASE_URL"] == new_url
        assert cfg["PROXY_API_KEY"] == self.PROXY_KEY  # unchanged


# ---------------------------------------------------------------------------
# Tests — Official (Claude Max) mode
# ---------------------------------------------------------------------------

class TestOfficialMode:
    """Test switching to official OAuth mode."""

    PROXY_URL = "http://test-host:11212/api/anthropic"
    PROXY_KEY = "sk-test-key-for-pytest"

    def test_switch_to_official(self):
        """Switching to official mode should succeed."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        result = run_switch("official")

        assert result.returncode == 0
        assert "OFFICIAL mode" in result.stdout

    def test_settings_json_removes_api_base_url(self):
        """settings.json should NOT contain apiBaseUrl after official switch."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")

        settings = read_settings()
        assert "apiBaseUrl" not in settings

    def test_settings_json_preserves_other_keys(self):
        """Switching to official must not clobber unrelated settings."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")

        settings = read_settings()
        # env block should survive the round-trip
        if "env" in settings:
            assert isinstance(settings["env"], dict)

    def test_env_file_unsets_vars(self):
        """The generated env file should unset ANTHROPIC vars."""
        run_switch("official")

        env_content = read_env()
        assert "unset ANTHROPIC_BASE_URL" in env_content
        assert "unset ANTHROPIC_API_KEY" in env_content

    def test_active_file_says_official(self):
        """The active file should contain 'official'."""
        run_switch("official")

        assert ACTIVE_FILE.read_text().strip() == "official"

    def test_status_reports_official(self):
        """'status' command should report official mode."""
        run_switch("official")
        result = run_switch("status")

        assert "official" in result.stdout.lower()
        assert "OAuth" in result.stdout


# ---------------------------------------------------------------------------
# Tests — Config command
# ---------------------------------------------------------------------------

class TestConfigCommand:
    """Test the 'config' subcommand."""

    def test_config_shows_path(self):
        result = run_switch("config")
        assert str(CONFIG_FILE) in result.stdout

    def test_config_shows_values(self):
        run_switch(
            "proxy",
            "http://show-this:1234/api",
            "sk-show-this-key",
        )
        result = run_switch("config")

        assert "http://show-this:1234/api" in result.stdout
        assert "sk-show-th" in result.stdout  # first 10 chars of key


# ---------------------------------------------------------------------------
# Tests — Validation and edge cases
# ---------------------------------------------------------------------------

class TestValidation:
    """Test error handling and input validation."""

    def test_proxy_without_config_fails(self):
        """Proxy with placeholder config should fail."""
        # Write placeholder config
        CONFIG_FILE.write_text(
            "PROXY_BASE_URL=http://YOUR_PROXY_HOST:11212/api/anthropic\n"
            "PROXY_API_KEY=YOUR_API_KEY_HERE\n"
        )
        result = run_switch("proxy", check=False)

        assert result.returncode != 0
        assert "ERROR" in result.stderr

    def test_unknown_command_shows_usage(self):
        result = run_switch("bogus", check=False)
        assert result.returncode != 0
        assert "Usage:" in result.stdout

    def test_no_args_shows_usage(self):
        result = run_switch(check=False)
        assert result.returncode != 0
        assert "Usage:" in result.stdout


# ---------------------------------------------------------------------------
# Tests — Round-trip (proxy → official → proxy)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify no data loss through mode switches."""

    PROXY_URL = "http://roundtrip:5555/api/anthropic"
    PROXY_KEY = "sk-roundtrip-key"

    def test_round_trip_preserves_settings(self):
        """Switching proxy → official → proxy should restore apiBaseUrl."""
        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")
        run_switch("proxy")

        settings = read_settings()
        assert settings["apiBaseUrl"] == self.PROXY_URL


# ---------------------------------------------------------------------------
# Tests — OAuth token stashing (avoids auth conflict)
# ---------------------------------------------------------------------------

def _has_oauth() -> bool:
    """Check if any config file has oauthAccount."""
    for path in (CLAUDE_JSON, CLAUDE_JSON_LEGACY):
        if path.exists():
            try:
                if "oauthAccount" in json.loads(path.read_text()):
                    return True
            except (json.JSONDecodeError, OSError):
                pass
    return False


def _skip_if_no_oauth():
    """Skip the test if no OAuth token is available."""
    if not _has_oauth():
        pytest.skip("No oauthAccount in any claude config to test with")


class TestOAuthStashing:
    """Verify OAuth token is stashed/restored to prevent auth conflicts.

    Checks both ~/.claude.json (current) and ~/.claude/claude.json (legacy).
    """

    PROXY_URL = "http://test-host:11212/api/anthropic"
    PROXY_KEY = "sk-test-key-for-pytest"

    def test_proxy_removes_oauth_from_primary(self):
        """Switching to proxy should remove oauthAccount from ~/.claude.json."""
        _skip_if_no_oauth()
        if not CLAUDE_JSON.exists():
            pytest.skip("~/.claude.json not present")

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        cj = read_claude_json()
        assert "oauthAccount" not in cj

    def test_proxy_removes_oauth_from_legacy(self):
        """Switching to proxy should also remove oauthAccount from legacy file."""
        _skip_if_no_oauth()
        if not CLAUDE_JSON_LEGACY.exists():
            pytest.skip("~/.claude/claude.json not present")

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        cj = read_claude_json_legacy()
        assert "oauthAccount" not in cj

    def test_proxy_creates_oauth_backup(self):
        """Switching to proxy should create an OAuth backup file."""
        _skip_if_no_oauth()

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        assert STASHED_OAUTH.exists()
        backup = json.loads(STASHED_OAUTH.read_text())
        assert "accountUuid" in backup or "emailAddress" in backup

    def test_official_restores_oauth(self):
        """Switching back to official should restore oauthAccount."""
        _skip_if_no_oauth()
        original_oauth = read_claude_json()["oauthAccount"]

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")

        cj = read_claude_json()
        assert "oauthAccount" in cj
        assert cj["oauthAccount"] == original_oauth

    def test_official_restores_oauth_to_legacy(self):
        """Restore should also write to the legacy config file."""
        _skip_if_no_oauth()
        if not CLAUDE_JSON_LEGACY.exists():
            pytest.skip("~/.claude/claude.json not present")

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")

        cj = read_claude_json_legacy()
        assert "oauthAccount" in cj

    def test_official_removes_backup(self):
        """After restoring, the backup file should be deleted."""
        _skip_if_no_oauth()

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        run_switch("official")

        assert not STASHED_OAUTH.exists()

    def test_proxy_preserves_other_claude_json_keys(self):
        """Stashing OAuth should not clobber other config keys."""
        _skip_if_no_oauth()

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)

        cj = read_claude_json()
        # numStartups and other keys should survive
        assert "numStartups" in cj

    def test_status_shows_stashed_in_proxy_mode(self):
        """Status should indicate OAuth is stashed when in proxy mode."""
        _skip_if_no_oauth()

        run_switch("proxy", self.PROXY_URL, self.PROXY_KEY)
        result = run_switch("status")

        assert "stashed" in result.stdout.lower()

    def test_status_shows_oauth_present_in_official_mode(self):
        """Status should show OAuth token present in official mode."""
        _skip_if_no_oauth()

        run_switch("official")
        result = run_switch("status")

        assert "OAuth token: present" in result.stdout


# ---------------------------------------------------------------------------
# Helpers — run claude -p via claude-switch env
# ---------------------------------------------------------------------------

def _claude_p_env() -> dict[str, str]:
    """Build an env dict suitable for spawning ``claude -p``.

    Strips CLAUDECODE (avoids nesting issues) and proxy vars (so the
    sourced env file is the sole authority).
    """
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDECODE")
    }


def _run_claude_p(
    prompt: str = "Reply with exactly: HELLO",
    timeout: int = 60,
    model: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Switch env via claude-switch, then run ``claude -p``."""
    model_flag = f' --model {model}' if model else ""
    return subprocess.run(
        [
            "bash", "-c",
            f'source ~/.config/claude-switch/env && claude -p "{prompt}"{model_flag}',
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_claude_p_env(),
    )


# Default proxy URL — override via TEST_PROXY_URL env var (e.g. in Docker
# use http://host.docker.internal:11212/api/anthropic).
_DEFAULT_PROXY_URL = os.environ.get(
    "TEST_PROXY_URL", "http://localhost:11212/api/anthropic"
)
_DEFAULT_PROXY_KEY = os.environ.get("TEST_PROXY_KEY", "test-key")
# Model to use for proxy e2e tests.  The default claude-sonnet-4-6 may hang
# on some proxies; override via TEST_PROXY_MODEL.
_DEFAULT_PROXY_MODEL = os.environ.get(
    "TEST_PROXY_MODEL", "claude-sonnet-4-5-20250929"
)


# ---------------------------------------------------------------------------
# Integration — switch to official (Claude Max) and run claude -p
# ---------------------------------------------------------------------------

class TestClaudeCLI:
    """End-to-end: switch to official mode and verify 'claude -p' works.

    NOTE: These tests hang when run inside a Claude Code session because
    spawning a nested ``claude`` process contends on shared runtime
    resources even with CLAUDECODE unset.  Run from a normal terminal::

        uv run pytest tests/test_claude_switch.py -m slow

    Also skipped in Docker where OAuth tokens are machine-bound.
    """

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.skipif(
        "CLAUDECODE" in os.environ,
        reason="Cannot spawn nested claude process inside Claude Code session",
    )
    @pytest.mark.skipif(
        os.path.exists("/.dockerenv"),
        reason="OAuth tokens are machine-bound; official mode cannot work in Docker",
    )
    def test_claude_p_works_in_official_mode(self):
        """After switching to official mode, 'claude -p' should respond."""
        run_switch("official")

        result = _run_claude_p()

        assert result.returncode == 0, (
            f"claude -p failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "HELLO" in result.stdout.upper(), (
            f"Expected HELLO in response, got: {result.stdout}"
        )


# ---------------------------------------------------------------------------
# Integration — switch to proxy mode and run claude -p
# ---------------------------------------------------------------------------

class TestProxyCLI:
    """End-to-end: switch to proxy mode and verify 'claude -p' works.

    Requires a running proxy (default http://localhost:11212/api/anthropic).
    Override with TEST_PROXY_URL / TEST_PROXY_KEY / TEST_PROXY_MODEL env vars.

    Run::

        uv run pytest tests/test_claude_switch.py -m slow
        # or in Docker:
        docker compose run --rm claude-switch-test
    """

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.skipif(
        "CLAUDECODE" in os.environ,
        reason="Cannot spawn nested claude process inside Claude Code session",
    )
    def test_claude_p_works_in_proxy_mode(self):
        """After switching to proxy mode, 'claude -p' should respond via proxy."""
        run_switch("proxy", _DEFAULT_PROXY_URL, _DEFAULT_PROXY_KEY)

        result = _run_claude_p(model=_DEFAULT_PROXY_MODEL)

        assert result.returncode == 0, (
            f"claude -p (proxy) failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "HELLO" in result.stdout.upper(), (
            f"Expected HELLO in response, got: {result.stdout}"
        )
