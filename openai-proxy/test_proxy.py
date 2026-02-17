"""Tests for the OpenAI-to-Claude Code CLI proxy."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from httpx import ASGITransport

from proxy import (
    app,
    build_claude_cmd,
    build_prompt,
    make_chunk,
    make_completion_response,
    resolve_model,
    DEFAULT_MODEL,
    CLAUDE_CLI,
)


# ---------------------------------------------------------------------------
# Unit tests: resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_maps_gpt4_to_sonnet(self):
        assert resolve_model("gpt-4") == "claude-sonnet-4-5-20250929"

    def test_maps_gpt4o_to_sonnet(self):
        assert resolve_model("gpt-4o") == "claude-sonnet-4-5-20250929"

    def test_maps_gpt35_to_haiku(self):
        assert resolve_model("gpt-3.5-turbo") == "claude-haiku-4-5-20251001"

    def test_maps_gpt4o_mini_to_haiku(self):
        assert resolve_model("gpt-4o-mini") == "claude-haiku-4-5-20251001"

    def test_maps_o3_to_opus(self):
        assert resolve_model("o3") == "claude-opus-4-6"

    def test_passes_through_claude_model(self):
        assert resolve_model("claude-opus-4-6") == "claude-opus-4-6"

    def test_passes_through_unknown_model(self):
        assert resolve_model("some-custom-model") == "some-custom-model"

    def test_none_returns_default(self):
        assert resolve_model(None) == DEFAULT_MODEL

    def test_empty_string_returns_default(self):
        assert resolve_model("") == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Unit tests: build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt, system = build_prompt(messages)
        assert prompt == "Human: Hello"
        assert system is None

    def test_system_message_extracted(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        prompt, system = build_prompt(messages)
        assert prompt == "Human: Hi"
        assert system == "You are helpful"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        prompt, system = build_prompt(messages)
        assert "Human: What is 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "Human: And 3+3?" in prompt

    def test_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "system", "content": "Use formal language"},
            {"role": "user", "content": "Hi"},
        ]
        prompt, system = build_prompt(messages)
        assert system == "Be concise\n\nUse formal language"

    def test_content_as_list_of_parts(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "text", "text": "and tell me"},
                ],
            }
        ]
        prompt, system = build_prompt(messages)
        assert "Look at this" in prompt
        assert "and tell me" in prompt

    def test_content_as_list_of_strings(self):
        messages = [
            {
                "role": "user",
                "content": ["Hello", "World"],
            }
        ]
        prompt, system = build_prompt(messages)
        assert "Hello" in prompt
        assert "World" in prompt

    def test_empty_messages(self):
        prompt, system = build_prompt([])
        assert prompt == ""
        assert system is None

    def test_system_only_returns_empty_prompt(self):
        messages = [{"role": "system", "content": "Be helpful"}]
        prompt, system = build_prompt(messages)
        assert prompt == ""
        assert system == "Be helpful"


# ---------------------------------------------------------------------------
# Unit tests: build_claude_cmd
# ---------------------------------------------------------------------------


class TestBuildClaudeCmd:
    def test_basic_command(self):
        cmd = build_claude_cmd("Hello")
        assert cmd[0] == CLAUDE_CLI
        assert "-p" in cmd
        assert "Hello" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_with_system_prompt(self):
        cmd = build_claude_cmd("Hello", system_prompt="Be helpful")
        assert "--append-system-prompt" in cmd
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Be helpful"

    def test_with_model(self):
        cmd = build_claude_cmd("Hello", model="claude-opus-4-6")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    def test_with_max_tokens(self):
        cmd = build_claude_cmd("Hello", max_tokens=500)
        assert "--max-tokens" in cmd
        idx = cmd.index("--max-tokens")
        assert cmd[idx + 1] == "500"

    def test_without_max_tokens(self):
        cmd = build_claude_cmd("Hello")
        assert "--max-tokens" not in cmd

    def test_streaming_format(self):
        cmd = build_claude_cmd("Hello", stream=True)
        assert "stream-json" in cmd
        assert "--verbose" in cmd

    def test_non_streaming_format(self):
        cmd = build_claude_cmd("Hello", stream=False)
        assert "json" in cmd
        assert "--verbose" not in cmd

    def test_with_session_id(self):
        cmd = build_claude_cmd("Hello", session_id="abc123")
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "abc123"

    def test_without_session_id(self):
        cmd = build_claude_cmd("Hello")
        assert "--session-id" not in cmd


# ---------------------------------------------------------------------------
# Unit tests: make_completion_response
# ---------------------------------------------------------------------------


class TestMakeCompletionResponse:
    def test_basic_response_structure(self):
        resp = make_completion_response("Hello!", "claude-opus-4-6")
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "claude-opus-4-6"
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert resp["choices"][0]["message"]["content"] == "Hello!"
        assert resp["choices"][0]["finish_reason"] == "stop"

    def test_id_contains_session_id(self):
        resp = make_completion_response("Hi", "m", session_id="sess123")
        assert resp["id"] == "chatcmpl-sess123"

    def test_id_generated_without_session(self):
        resp = make_completion_response("Hi", "m")
        assert resp["id"].startswith("chatcmpl-")

    def test_custom_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        resp = make_completion_response("Hi", "m", usage=usage)
        assert resp["usage"] == usage

    def test_default_usage_zeros(self):
        resp = make_completion_response("Hi", "m")
        assert resp["usage"]["prompt_tokens"] == 0
        assert resp["usage"]["completion_tokens"] == 0
        assert resp["usage"]["total_tokens"] == 0

    def test_created_is_recent_timestamp(self):
        resp = make_completion_response("Hi", "m")
        assert abs(resp["created"] - int(time.time())) <= 2


# ---------------------------------------------------------------------------
# Unit tests: make_chunk
# ---------------------------------------------------------------------------


class TestMakeChunk:
    def test_basic_chunk(self):
        chunk = make_chunk("Hello", "m", "id1")
        assert chunk["id"] == "id1"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_chunk_with_role(self):
        chunk = make_chunk("Hello", "m", "id1", include_role=True)
        assert chunk["choices"][0]["delta"]["role"] == "assistant"
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_chunk_without_role(self):
        chunk = make_chunk("Hello", "m", "id1", include_role=False)
        assert "role" not in chunk["choices"][0]["delta"]

    def test_finish_chunk(self):
        chunk = make_chunk("", "m", "id1", finish_reason="stop")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert "content" not in chunk["choices"][0]["delta"]

    def test_empty_content_not_in_delta(self):
        chunk = make_chunk("", "m", "id1")
        assert "content" not in chunk["choices"][0]["delta"]


# ---------------------------------------------------------------------------
# Integration tests: HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_models_endpoint(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    # Claude models present
    assert "claude-opus-4-6" in model_ids
    assert "claude-sonnet-4-5-20250929" in model_ids
    assert "claude-haiku-4-5-20251001" in model_ids
    # OpenAI aliases present
    assert "gpt-4" in model_ids
    assert "gpt-3.5-turbo" in model_ids


@pytest.mark.asyncio
async def test_chat_completions_empty_messages(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": []},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_chat_completions_system_only_returns_400(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "Be helpful"}],
        },
    )
    assert resp.status_code == 400


class AsyncLineIterator:
    """Async iterator over lines, simulating asyncio.subprocess stdout."""

    def __init__(self, lines: list[bytes]):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


def _mock_process(stdout_data: bytes, stderr_data: bytes = b"", returncode: int = 0):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout_data, stderr_data))
    proc.wait = AsyncMock(return_value=returncode)

    # Mock stdout as async iterator for streaming
    lines = [line + b"\n" for line in stdout_data.split(b"\n") if line]
    proc.stdout = AsyncLineIterator(lines)

    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr_data)
    return proc


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(client):
    """Test non-streaming chat completion with mocked Claude CLI."""
    cli_response = json.dumps({
        "result": "Hello! How can I help?",
        "session_id": "test-session-123",
        "num_input_tokens": 10,
        "num_output_tokens": 8,
    }).encode()

    proc = _mock_process(cli_response)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 8
    assert data["usage"]["total_tokens"] == 18
    assert data["id"] == "chatcmpl-test-session-123"
    assert resp.headers.get("x-session-id") == "test-session-123"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_raw_text(client):
    """Test fallback when CLI returns non-JSON text."""
    proc = _mock_process(b"Just plain text response")

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Just plain text response"


@pytest.mark.asyncio
async def test_chat_completions_cli_error(client):
    """Test error handling when Claude CLI exits non-zero."""
    proc = _mock_process(b"", b"Error: something went wrong", returncode=1)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 502
    assert "upstream_error" in resp.json()["error"]["type"]


@pytest.mark.asyncio
async def test_chat_completions_cli_not_found(client):
    """Test error handling when Claude CLI binary doesn't exist."""
    with patch(
        "proxy.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("No such file"),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 500
    assert "configuration_error" in resp.json()["error"]["type"]


@pytest.mark.asyncio
async def test_chat_completions_with_session_header(client):
    """Test that x-session-id header is passed through."""
    cli_response = json.dumps({
        "result": "Continued conversation",
        "session_id": "existing-session",
    }).encode()

    proc = _mock_process(cli_response)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Continue"}],
            },
            headers={"x-session-id": "existing-session"},
        )

    assert resp.status_code == 200
    # Verify session-id was passed to CLI
    call_args = mock_exec.call_args
    cmd_list = list(call_args[0])
    assert "--session-id" in cmd_list
    idx = cmd_list.index("--session-id")
    assert cmd_list[idx + 1] == "existing-session"


@pytest.mark.asyncio
async def test_chat_completions_with_system_prompt(client):
    """Test that system messages are extracted and passed via --append-system-prompt."""
    cli_response = json.dumps({"result": "OK"}).encode()
    proc = _mock_process(cli_response)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a pirate"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    assert resp.status_code == 200
    cmd_list = list(mock_exec.call_args[0])
    assert "--append-system-prompt" in cmd_list
    idx = cmd_list.index("--append-system-prompt")
    assert cmd_list[idx + 1] == "You are a pirate"


@pytest.mark.asyncio
async def test_chat_completions_with_max_tokens(client):
    """Test that max_tokens is forwarded to CLI."""
    cli_response = json.dumps({"result": "OK"}).encode()
    proc = _mock_process(cli_response)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
            },
        )

    assert resp.status_code == 200
    cmd_list = list(mock_exec.call_args[0])
    assert "--max-tokens" in cmd_list
    idx = cmd_list.index("--max-tokens")
    assert cmd_list[idx + 1] == "256"


@pytest.mark.asyncio
async def test_chat_completions_streaming(client):
    """Test streaming chat completion with mocked Claude CLI."""
    ndjson_lines = [
        json.dumps({"type": "system", "session_id": "stream-sess-1"}).encode(),
        json.dumps({"type": "content_block_delta", "delta": {"text": "Hello"}}).encode(),
        json.dumps({"type": "content_block_delta", "delta": {"text": " world"}}).encode(),
        json.dumps({"type": "result", "result": "Hello world"}).encode(),
    ]
    stdout_data = b"\n".join(ndjson_lines)

    proc = _mock_process(stdout_data, returncode=0)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Parse SSE events from body
    body = resp.text
    events = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(payload))

    # Should have content chunks
    content_chunks = [
        e for e in events
        if isinstance(e, dict) and e.get("choices", [{}])[0].get("delta", {}).get("content")
    ]
    assert len(content_chunks) >= 1

    # First content chunk should include role
    assert content_chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Last event should be [DONE]
    assert events[-1] == "[DONE]"

    # Second-to-last should be finish chunk with stop
    finish_events = [
        e for e in events
        if isinstance(e, dict) and e.get("choices", [{}])[0].get("finish_reason") == "stop"
    ]
    assert len(finish_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_streaming_with_assistant_type(client):
    """Test streaming handles 'assistant' type events."""
    ndjson_lines = [
        json.dumps({"type": "assistant", "message": "Hi there"}).encode(),
    ]
    stdout_data = b"\n".join(ndjson_lines)

    proc = _mock_process(stdout_data, returncode=0)

    with patch("proxy.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    body = resp.text
    content_found = False
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload != "[DONE]":
                event = json.loads(payload)
                content = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if "Hi there" in content:
                    content_found = True
    assert content_found


@pytest.mark.asyncio
async def test_models_all_have_required_fields(client):
    """Verify each model entry has the fields OpenAI clients expect."""
    resp = await client.get("/v1/models")
    data = resp.json()
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model
