"""Tests for the OpenAI API-compatible Claude proxy."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import openai_proxy
from openai_proxy import (
    _extract_cli_text,
    _map_stop_reason,
    _messages_to_prompt_string,
    _resolve_model,
    _translate_messages,
    app,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons between tests."""
    openai_proxy._anthropic_client = None
    openai_proxy._CLI_ENV = None
    yield
    openai_proxy._anthropic_client = None
    openai_proxy._CLI_ENV = None


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_none_returns_default(self):
        assert _resolve_model(None) == openai_proxy.DEFAULT_MODEL

    def test_empty_string_returns_default(self):
        assert _resolve_model("") == openai_proxy.DEFAULT_MODEL

    def test_claude_passthrough(self):
        assert _resolve_model("claude-opus-4-6") == "claude-opus-4-6"
        assert _resolve_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"
        assert _resolve_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"

    def test_gpt4_maps_to_sonnet(self):
        assert _resolve_model("gpt-4") == "claude-sonnet-4-5-20250929"
        assert _resolve_model("gpt-4o") == "claude-sonnet-4-5-20250929"
        assert _resolve_model("gpt-4-turbo") == "claude-sonnet-4-5-20250929"

    def test_gpt35_maps_to_haiku(self):
        assert _resolve_model("gpt-3.5-turbo") == "claude-haiku-4-5-20251001"

    def test_gpt4o_mini_maps_to_haiku(self):
        assert _resolve_model("gpt-4o-mini") == "claude-haiku-4-5-20251001"

    def test_o1_maps_to_opus(self):
        assert _resolve_model("o1") == "claude-opus-4-6"
        assert _resolve_model("o1-preview") == "claude-opus-4-6"

    def test_unknown_model_returns_default(self):
        assert _resolve_model("some-unknown-model") == openai_proxy.DEFAULT_MODEL


class TestMapStopReason:
    def test_end_turn(self):
        assert _map_stop_reason("end_turn") == "stop"

    def test_max_tokens(self):
        assert _map_stop_reason("max_tokens") == "length"

    def test_stop_sequence(self):
        assert _map_stop_reason("stop_sequence") == "stop"

    def test_none(self):
        assert _map_stop_reason(None) == "stop"

    def test_unknown(self):
        assert _map_stop_reason("something_else") == "stop"


class TestTranslateMessages:
    def test_simple_user_message(self):
        system, msgs = _translate_messages([{"role": "user", "content": "Hello"}])
        assert system is None
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_system_message_extracted(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "You are helpful."
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_multiple_system_messages_joined(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "Be helpful.\n\nBe concise."
        assert len(msgs) == 1

    def test_consecutive_user_messages_merged(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ])
        assert system is None
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello\nWorld"

    def test_consecutive_assistant_messages_merged(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "How can I help?"},
        ])
        assert len(msgs) == 2
        assert msgs[1]["content"] == "Hello\nHow can I help?"

    def test_alternating_roles_preserved(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_assistant_first_gets_user_prefix(self):
        system, msgs = _translate_messages([
            {"role": "assistant", "content": "I started"},
        ])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "(continued conversation)"
        assert msgs[1]["role"] == "assistant"

    def test_empty_messages_get_fallback(self):
        system, msgs = _translate_messages([])
        assert system is None
        assert msgs == [{"role": "user", "content": "(empty)"}]

    def test_non_string_system_content_serialized(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": ["structured", "data"]},
            {"role": "user", "content": "Hi"},
        ])
        assert system == json.dumps(["structured", "data"])

    def test_function_role_becomes_user(self):
        """Non-user/non-assistant roles like 'function' or 'tool' become 'user'."""
        system, msgs = _translate_messages([
            {"role": "user", "content": "Use a tool"},
            {"role": "assistant", "content": "Calling tool..."},
            {"role": "tool", "content": "tool result"},
        ])
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "tool result"


class TestExtractCliText:
    def test_content_block_delta(self):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}
        assert _extract_cli_text(event) == "hello"

    def test_result_event(self):
        event = {"type": "result", "result": "final answer"}
        assert _extract_cli_text(event) == "final answer"

    def test_assistant_string_message(self):
        event = {"type": "assistant", "message": "hi"}
        assert _extract_cli_text(event) == "hi"

    def test_assistant_dict_message(self):
        event = {"type": "assistant", "message": {"content": [{"type": "text", "text": "hey"}]}}
        assert _extract_cli_text(event) == "hey"

    def test_unknown_event_returns_none(self):
        assert _extract_cli_text({"type": "ping"}) is None

    def test_content_block_delta_wrong_subtype(self):
        event = {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}}
        assert _extract_cli_text(event) is None


class TestMessagesToPromptString:
    def test_simple_messages(self):
        result = _messages_to_prompt_string([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])
        assert "[System]: Be helpful" in result
        assert "[User]: Hello" in result
        assert "[Assistant]: Hi there" in result

    def test_list_content_flattened(self):
        result = _messages_to_prompt_string([
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]},
        ])
        assert "[User]: Hello World" in result

    def test_empty_messages(self):
        assert _messages_to_prompt_string([]) == ""


# ---------------------------------------------------------------------------
# Route tests: health and models
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "backend" in data


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 3

    def test_model_format(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "anthropic"
        assert "created" in model

    def test_all_models_present(self, client):
        resp = client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert "claude-opus-4-6" in ids
        assert "claude-sonnet-4-5-20250929" in ids
        assert "claude-haiku-4-5-20251001" in ids


# ---------------------------------------------------------------------------
# Route tests: chat completions (Anthropic backend, mocked)
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str = "Hello!", stop_reason: str = "end_turn"):
    """Build a mock Anthropic Messages response."""
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=usage)


class TestChatCompletionsNonStreaming:
    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_basic_completion(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("Hi there!")
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hi there!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_usage_included(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_system_message_forwarded(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_temperature_forwarded(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_default_max_tokens(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_custom_max_tokens(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_model_name_mapping(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert resp.json()["model"] == "claude-sonnet-4-5-20250929"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_max_tokens_stop_reason(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("truncated", stop_reason="max_tokens")
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.json()["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# Route tests: chat completions streaming (Anthropic backend, mocked)
# ---------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_get_client, async_client):
        # Build a mock async stream context manager
        async def mock_text_stream():
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        mock_stream = MagicMock()
        mock_stream.text_stream = mock_text_stream()

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream_cm)
        mock_get_client.return_value = mock_client

        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        body = resp.text
        data_lines = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ")
        ]

        # Should have content chunks + final + [DONE]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "[DONE]"

        # Check a content chunk
        first_chunk = json.loads(data_lines[0])
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "delta" in first_chunk["choices"][0]

        # Check final chunk has finish_reason
        final_chunk = json.loads(data_lines[-2])
        assert final_chunk["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Route tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_json(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_auth_error(self, mock_get_client, client):
        import httpx

        mock_client = MagicMock()
        mock_response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        mock_client.messages.create = AsyncMock(
            side_effect=openai_proxy.anthropic.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body=None,
            )
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_generic_exception(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=RuntimeError("something broke")
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "server_error"
        assert "something broke" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Route tests: CLI backend (mocked subprocess)
# ---------------------------------------------------------------------------


class TestCLIBackendNonStreaming:
    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_success(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(
                json.dumps({"result": "Hello from CLI!"}).encode(),
                b"",
            )
        )
        mock_exec.return_value = mock_proc

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from CLI!"
        assert data["usage"]["prompt_tokens"] == 0

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_error(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"CLI failed")
        )
        mock_exec.return_value = mock_proc

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 502
        assert "CLI error" in resp.json()["error"]["message"]

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_plain_text_output(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"Plain text response", b"")
        )
        mock_exec.return_value = mock_proc

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Plain text response"

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_excludes_claudecode_env(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b"")
        )
        mock_exec.return_value = mock_proc

        with patch.dict("os.environ", {"CLAUDECODE": "1", "HOME": "/home/test"}, clear=False):
            client.post("/v1/chat/completions", json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
            })

        call_kwargs = mock_exec.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert "CLAUDECODE" not in env


class TestCLIBackendStreaming:
    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_cli_streaming(self, mock_exec, async_client):
        lines = [
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}) + "\n",
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}) + "\n",
        ]

        async def fake_readline():
            if lines:
                return lines.pop(0).encode()
            return b""

        mock_stdout = MagicMock()
        mock_stdout.__aiter__ = lambda self: self
        _lines = iter([
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}).encode() + b"\n",
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}).encode() + b"\n",
        ])
        mock_stdout.__anext__ = AsyncMock(side_effect=lambda: next(_lines, StopAsyncIteration()))

        # Need proper async iteration
        async def aiter_lines():
            yield json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}).encode() + b"\n"
            yield json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}).encode() + b"\n"

        mock_proc = MagicMock()
        mock_proc.stdout = aiter_lines()
        mock_proc.stderr = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        assert data_lines[-1] == "[DONE]"

        # Check content was streamed
        content_chunks = []
        for dl in data_lines:
            if dl == "[DONE]":
                break
            parsed = json.loads(dl)
            delta_content = parsed["choices"][0]["delta"].get("content")
            if delta_content:
                content_chunks.append(delta_content)
        assert "Hello" in content_chunks
        assert " world" in content_chunks

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_cli_streaming_result_event(self, mock_exec, async_client):
        """Test that 'result' type events from CLI are captured."""
        async def aiter_lines():
            yield json.dumps({"type": "result", "result": "Final answer"}).encode() + b"\n"

        mock_proc = MagicMock()
        mock_proc.stdout = aiter_lines()
        mock_proc.stderr = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_exec.return_value = mock_proc

        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        content_chunks = []
        for dl in data_lines:
            if dl == "[DONE]":
                break
            parsed = json.loads(dl)
            c = parsed["choices"][0]["delta"].get("content")
            if c:
                content_chunks.append(c)
        assert "Final answer" in content_chunks


# ---------------------------------------------------------------------------
# Edge case / integration-level tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_no_model_field_uses_default(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == openai_proxy.DEFAULT_MODEL

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_empty_messages_handled(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [],
        })

        assert resp.status_code == 200
        call_kwargs = mock_client.messages.create.call_args[1]
        assert len(call_kwargs["messages"]) >= 1

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_no_system_kwarg_when_no_system_message(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_no_temperature_kwarg_when_not_set(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_multi_block_response(self, mock_get_client, client):
        """Test that multiple text blocks are concatenated."""
        block1 = SimpleNamespace(type="text", text="Hello ")
        block2 = SimpleNamespace(type="text", text="world!")
        block3 = SimpleNamespace(type="tool_use", id="x", name="t", input={})
        usage = SimpleNamespace(input_tokens=5, output_tokens=3)
        response = SimpleNamespace(
            content=[block1, block2, block3],
            stop_reason="end_turn",
            usage=usage,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.json()["choices"][0]["message"]["content"] == "Hello world!"
