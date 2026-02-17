"""Tests for the OpenAI API-compatible Claude proxy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import openai_proxy
from openai_proxy import (
    _build_anthropic_kwargs,
    _build_cli_cmd,
    _extract_cli_text,
    _make_chunk,
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
    yield
    openai_proxy._anthropic_client = None


@pytest.fixture(autouse=True)
def _disable_auth():
    """Disable auth for all tests by default."""
    with patch.object(openai_proxy, "API_KEY", ""):
        yield


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Unit tests: _resolve_model
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

    def test_o3_maps_to_opus(self):
        assert _resolve_model("o3") == "claude-opus-4-6"

    def test_unknown_model_passes_through(self):
        assert _resolve_model("some-custom-model") == "some-custom-model"


# ---------------------------------------------------------------------------
# Unit tests: _map_stop_reason
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unit tests: _translate_messages
# ---------------------------------------------------------------------------


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

    def test_list_system_content_flattened(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": ["structured", "data"]},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "structured\ndata"

    def test_tool_role_becomes_user(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "Use a tool"},
            {"role": "assistant", "content": "Calling tool..."},
            {"role": "tool", "content": "tool result"},
        ])
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "tool result"

    def test_list_content_parts_flattened(self):
        system, msgs = _translate_messages([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "text", "text": "and tell me"},
                ],
            }
        ])
        assert "Look at this" in msgs[0]["content"]
        assert "and tell me" in msgs[0]["content"]

    def test_list_content_with_plain_strings(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": ["Hello", "World"]},
        ])
        assert "Hello" in msgs[0]["content"]
        assert "World" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Unit tests: _build_anthropic_kwargs
# ---------------------------------------------------------------------------


class TestBuildAnthropicKwargs:
    def test_basic_kwargs(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert kwargs["max_tokens"] == 4096
        assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert "system" not in kwargs
        assert "temperature" not in kwargs

    def test_system_included(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["system"] == "Be helpful"

    def test_temperature_included(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["temperature"] == 0.5

    def test_custom_max_tokens(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 200,
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["max_tokens"] == 200

    def test_model_mapping(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["model"] == "claude-sonnet-4-5-20250929"

    def test_no_model_uses_default(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["model"] == openai_proxy.DEFAULT_MODEL

    def test_top_p_forwarded(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 0.9,
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["top_p"] == 0.9

    def test_stop_string_converted_to_list(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["stop_sequences"] == ["END"]

    def test_stop_list_forwarded(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "STOP"],
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["stop_sequences"] == ["END", "STOP"]

    def test_max_tokens_zero_preserved(self):
        """max_tokens=0 should not be replaced by the default."""
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        }
        kwargs = _build_anthropic_kwargs(body)
        assert kwargs["max_tokens"] == 0


# ---------------------------------------------------------------------------
# Unit tests: _extract_cli_text
# ---------------------------------------------------------------------------


class TestExtractCliText:
    def test_content_block_delta(self):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}
        assert _extract_cli_text(event) == "hello"

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

    def test_result_event_returns_none(self):
        """Result events should not emit text to avoid double-emit."""
        assert _extract_cli_text({"type": "result", "result": "final"}) is None


# ---------------------------------------------------------------------------
# Unit tests: _messages_to_prompt_string
# ---------------------------------------------------------------------------


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
# Unit tests: _build_cli_cmd
# ---------------------------------------------------------------------------


class TestBuildCliCmd:
    def test_basic_command(self):
        cmd = _build_cli_cmd("Hello")
        assert cmd[0] == openai_proxy.CLAUDE_CLI
        assert "-p" in cmd
        assert "Hello" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_with_system_prompt(self):
        cmd = _build_cli_cmd("Hello", system_prompt="Be helpful")
        assert "--append-system-prompt" in cmd
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Be helpful"

    def test_with_model(self):
        cmd = _build_cli_cmd("Hello", model="claude-opus-4-6")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    def test_with_max_tokens(self):
        cmd = _build_cli_cmd("Hello", max_tokens=500)
        assert "--max-tokens" in cmd
        idx = cmd.index("--max-tokens")
        assert cmd[idx + 1] == "500"

    def test_without_max_tokens(self):
        cmd = _build_cli_cmd("Hello")
        assert "--max-tokens" not in cmd

    def test_streaming_format(self):
        cmd = _build_cli_cmd("Hello", stream=True)
        assert "stream-json" in cmd
        assert "--verbose" in cmd

    def test_non_streaming_format(self):
        cmd = _build_cli_cmd("Hello", stream=False)
        assert "json" in cmd
        assert "--verbose" not in cmd

    def test_with_session_id(self):
        cmd = _build_cli_cmd("Hello", session_id="abc123")
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "abc123"

    def test_without_session_id(self):
        cmd = _build_cli_cmd("Hello")
        assert "--session-id" not in cmd


# ---------------------------------------------------------------------------
# Unit tests: _make_chunk
# ---------------------------------------------------------------------------


class TestMakeChunk:
    def test_content_chunk(self):
        chunk = _make_chunk("id1", 100, "model", content="Hello")
        assert chunk["id"] == "id1"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_finish_chunk(self):
        chunk = _make_chunk("id1", 100, "model", finish_reason="stop")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert "content" not in chunk["choices"][0]["delta"]

    def test_empty_content_not_in_delta(self):
        chunk = _make_chunk("id1", 100, "model")
        assert "content" not in chunk["choices"][0]["delta"]

    def test_usage_included(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        chunk = _make_chunk("id1", 100, "model", usage=usage)
        assert chunk["usage"] == usage

    def test_no_usage_by_default(self):
        chunk = _make_chunk("id1", 100, "model")
        assert "usage" not in chunk


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
        assert len(data["data"]) > 3  # Claude models + aliases

    def test_model_format(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "anthropic"
        assert "created" in model

    def test_all_claude_models_present(self, client):
        resp = client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert "claude-opus-4-6" in ids
        assert "claude-sonnet-4-5-20250929" in ids
        assert "claude-haiku-4-5-20251001" in ids

    def test_openai_aliases_present(self, client):
        resp = client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert "gpt-4" in ids
        assert "gpt-3.5-turbo" in ids


# ---------------------------------------------------------------------------
# Auth middleware tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    def test_no_auth_required_when_key_unset(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_health_always_accessible(self):
        with patch.object(openai_proxy, "API_KEY", "secret123"):
            c = TestClient(app)
            resp = c.get("/health")
            assert resp.status_code == 200

    def test_rejects_missing_key(self):
        with patch.object(openai_proxy, "API_KEY", "secret123"):
            c = TestClient(app)
            resp = c.get("/v1/models")
            assert resp.status_code == 401

    def test_rejects_wrong_key(self):
        with patch.object(openai_proxy, "API_KEY", "secret123"):
            c = TestClient(app)
            resp = c.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrongkey"},
            )
            assert resp.status_code == 401

    def test_accepts_correct_key(self):
        with patch.object(openai_proxy, "API_KEY", "secret123"):
            c = TestClient(app)
            resp = c.get(
                "/v1/models",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Anthropic backend: non-streaming (mocked)
# ---------------------------------------------------------------------------


def _mock_response(
    text: str = "Hello!",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=usage)


class TestAnthropicNonStreaming:
    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_basic_completion(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response("Hi!"))
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hi!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_usage_included(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        usage = resp.json()["usage"]
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_system_message_forwarded(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
    def test_top_p_forwarded(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 0.9,
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_stop_sequences_forwarded(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END"],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["END"]

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_default_max_tokens(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
            return_value=_mock_response("truncated", stop_reason="max_tokens")
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.json()["choices"][0]["finish_reason"] == "length"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_multi_block_response(self, mock_get_client, client):
        """Multiple text blocks are concatenated."""
        b1 = SimpleNamespace(type="text", text="Hello ")
        b2 = SimpleNamespace(type="text", text="world!")
        b3 = SimpleNamespace(type="tool_use", id="x", name="t", input={})
        usage = SimpleNamespace(input_tokens=5, output_tokens=3)
        response = SimpleNamespace(
            content=[b1, b2, b3], stop_reason="end_turn", usage=usage
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.json()["choices"][0]["message"]["content"] == "Hello world!"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_no_model_field_uses_default(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
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
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs


# ---------------------------------------------------------------------------
# Anthropic backend: streaming (mocked)
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    @pytest.mark.asyncio
    async def test_streaming_chunks_and_done(self, mock_get_client, async_client):
        async def mock_text_stream():
            for chunk in ["Hello", " world"]:
                yield chunk

        final_msg = _mock_response("Hello world", input_tokens=8, output_tokens=4)

        mock_stream = MagicMock()
        mock_stream.text_stream = mock_text_stream()
        mock_stream.get_final_message = AsyncMock(return_value=final_msg)

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

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        assert data_lines[-1] == "[DONE]"

        # Content chunks
        content_chunks = []
        for dl in data_lines:
            if dl == "[DONE]":
                break
            parsed = json.loads(dl)
            c = parsed["choices"][0]["delta"].get("content")
            if c:
                content_chunks.append(c)
        assert "Hello" in content_chunks
        assert " world" in content_chunks

        # Final chunk has finish_reason and usage from get_final_message()
        final_chunk = json.loads(data_lines[-2])
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert final_chunk["usage"]["prompt_tokens"] == 8
        assert final_chunk["usage"]["completion_tokens"] == 4
        assert final_chunk["usage"]["total_tokens"] == 12

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    @pytest.mark.asyncio
    async def test_streaming_stop_reason_from_final_message(self, mock_get_client, async_client):
        async def mock_text_stream():
            yield "partial"

        final_msg = _mock_response("partial", stop_reason="max_tokens")
        mock_stream = MagicMock()
        mock_stream.text_stream = mock_text_stream()
        mock_stream.get_final_message = AsyncMock(return_value=final_msg)

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

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        final_chunk = json.loads(data_lines[-2])
        assert final_chunk["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# Error handling
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
    def test_rate_limit_error(self, mock_get_client, client):
        import httpx

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=openai_proxy.anthropic.RateLimitError(
                message="Rate limited",
                response=httpx.Response(
                    status_code=429,
                    request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
                ),
                body=None,
            )
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "rate_limit_error"

    @patch.object(openai_proxy, "BACKEND", "anthropic")
    @patch.object(openai_proxy, "_get_anthropic_client")
    def test_generic_exception_no_leak(self, mock_get_client, client):
        """Internal errors should not leak exception details."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=RuntimeError("secret internal path /etc/foo")
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "server_error"
        # Should NOT contain the internal error details
        assert "secret internal path" not in resp.json()["error"]["message"]
        assert resp.json()["error"]["message"] == "Internal server error"


# ---------------------------------------------------------------------------
# CLI backend: non-streaming (mocked)
# ---------------------------------------------------------------------------


class TestCLINonStreaming:
    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_success(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(
                json.dumps({
                    "result": "Hello from CLI!",
                    "session_id": "sess-123",
                    "num_input_tokens": 10,
                    "num_output_tokens": 8,
                }).encode(),
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
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 8
        assert data["usage"]["total_tokens"] == 18
        assert data["id"] == "chatcmpl-sess-123"
        assert resp.headers.get("x-session-id") == "sess-123"

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

        with patch.dict("os.environ", {"CLAUDECODE": "1"}, clear=False):
            client.post("/v1/chat/completions", json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
            })

        env = mock_exec.call_args.kwargs.get("env") or mock_exec.call_args[1].get("env")
        assert "CLAUDECODE" not in env

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No such file"))
    def test_cli_not_found(self, mock_exec, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "configuration_error"

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_session_id_header(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b"")
        )
        mock_exec.return_value = mock_proc

        client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"x-session-id": "my-session"},
        )

        cmd_list = list(mock_exec.call_args[0])
        assert "--session-id" in cmd_list
        idx = cmd_list.index("--session-id")
        assert cmd_list[idx + 1] == "my-session"

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_system_prompt_forwarded(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b"")
        )
        mock_exec.return_value = mock_proc

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "system", "content": "You are a pirate"},
                {"role": "user", "content": "Hello"},
            ],
        })

        cmd_list = list(mock_exec.call_args[0])
        assert "--append-system-prompt" in cmd_list
        idx = cmd_list.index("--append-system-prompt")
        assert cmd_list[idx + 1] == "You are a pirate"

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    def test_cli_max_tokens_forwarded(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b"")
        )
        mock_exec.return_value = mock_proc

        client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 256,
        })

        cmd_list = list(mock_exec.call_args[0])
        assert "--max-tokens" in cmd_list
        idx = cmd_list.index("--max-tokens")
        assert cmd_list[idx + 1] == "256"


# ---------------------------------------------------------------------------
# CLI backend: streaming (mocked)
# ---------------------------------------------------------------------------


class TestCLIStreaming:
    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_cli_streaming(self, mock_exec, async_client):
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
    async def test_cli_streaming_assistant_event(self, mock_exec, async_client):
        async def aiter_lines():
            yield json.dumps({"type": "assistant", "message": "Hi there"}).encode() + b"\n"

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

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        content_found = False
        for dl in data_lines:
            if dl == "[DONE]":
                break
            parsed = json.loads(dl)
            c = parsed["choices"][0]["delta"].get("content", "")
            if "Hi there" in c:
                content_found = True
        assert content_found

    @patch.object(openai_proxy, "BACKEND", "cli")
    @patch("openai_proxy.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_cli_streaming_session_id_from_system_event(self, mock_exec, async_client):
        async def aiter_lines():
            yield json.dumps({"type": "system", "session_id": "stream-sess-1"}).encode() + b"\n"
            yield json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}).encode() + b"\n"

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

        data_lines = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]

        # Content chunk should use session-based ID
        for dl in data_lines:
            if dl == "[DONE]":
                continue
            parsed = json.loads(dl)
            if parsed["choices"][0]["delta"].get("content"):
                assert parsed["id"] == "chatcmpl-stream-sess-1"
                break


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingletonClient:
    def test_anthropic_client_reused(self):
        with patch("openai_proxy.anthropic.AsyncAnthropic") as MockClass:
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance

            c1 = openai_proxy._get_anthropic_client()
            c2 = openai_proxy._get_anthropic_client()

            assert c1 is c2
            MockClass.assert_called_once()

    def test_cli_env_not_cached(self):
        """_get_cli_env computes fresh each call (no stale cache)."""
        e1 = openai_proxy._get_cli_env()
        e2 = openai_proxy._get_cli_env()
        assert e1 is not e2
        assert e1 == e2
