"""Tests for the OpenAI API-compatible Claude proxy (v2 - SDK optimized)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import openai_proxy_v2
from openai_proxy_v2 import (
    _build_anthropic_kwargs,
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
    openai_proxy_v2._anthropic_client = None
    openai_proxy_v2._CLI_ENV = None
    yield
    openai_proxy_v2._anthropic_client = None
    openai_proxy_v2._CLI_ENV = None


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
        assert _resolve_model(None) == openai_proxy_v2.DEFAULT_MODEL

    def test_empty_string_returns_default(self):
        assert _resolve_model("") == openai_proxy_v2.DEFAULT_MODEL

    def test_claude_passthrough(self):
        assert _resolve_model("claude-opus-4-6") == "claude-opus-4-6"
        assert _resolve_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"

    def test_gpt4_maps_to_sonnet(self):
        assert _resolve_model("gpt-4") == "claude-sonnet-4-5-20250929"
        assert _resolve_model("gpt-4o") == "claude-sonnet-4-5-20250929"

    def test_gpt35_maps_to_haiku(self):
        assert _resolve_model("gpt-3.5-turbo") == "claude-haiku-4-5-20251001"

    def test_o1_maps_to_opus(self):
        assert _resolve_model("o1") == "claude-opus-4-6"

    def test_unknown_model_returns_default(self):
        assert _resolve_model("unknown") == openai_proxy_v2.DEFAULT_MODEL


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
        assert _map_stop_reason("other") == "stop"


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
        system, _ = _translate_messages([
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "A\n\nB"

    def test_consecutive_same_role_merged(self):
        _, msgs = _translate_messages([
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello\nWorld"

    def test_assistant_first_gets_user_prefix(self):
        _, msgs = _translate_messages([{"role": "assistant", "content": "I started"}])
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_empty_messages_get_fallback(self):
        system, msgs = _translate_messages([])
        assert system is None
        assert msgs == [{"role": "user", "content": "(empty)"}]

    def test_tool_role_becomes_user(self):
        _, msgs = _translate_messages([
            {"role": "user", "content": "Use tool"},
            {"role": "assistant", "content": "OK"},
            {"role": "tool", "content": "result"},
        ])
        assert msgs[-1]["role"] == "user"


class TestBuildAnthropicKwargs:
    """Tests for the shared request builder (new in v2)."""

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
        assert kwargs["model"] == openai_proxy_v2.DEFAULT_MODEL


class TestExtractCliText:
    def test_content_block_delta(self):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}
        assert _extract_cli_text(event) == "hello"

    def test_result_event(self):
        assert _extract_cli_text({"type": "result", "result": "final"}) == "final"

    def test_assistant_string_message(self):
        assert _extract_cli_text({"type": "assistant", "message": "hi"}) == "hi"

    def test_assistant_dict_message(self):
        event = {"type": "assistant", "message": {"content": [{"type": "text", "text": "hey"}]}}
        assert _extract_cli_text(event) == "hey"

    def test_unknown_returns_none(self):
        assert _extract_cli_text({"type": "ping"}) is None

    def test_wrong_delta_subtype_returns_none(self):
        event = {"type": "content_block_delta", "delta": {"type": "input_json_delta"}}
        assert _extract_cli_text(event) is None


class TestMessagesToPromptString:
    def test_all_roles(self):
        result = _messages_to_prompt_string([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ])
        assert "[System]: sys" in result
        assert "[User]: usr" in result
        assert "[Assistant]: asst" in result

    def test_list_content_flattened(self):
        result = _messages_to_prompt_string([
            {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]},
        ])
        assert "[User]: A B" in result

    def test_empty(self):
        assert _messages_to_prompt_string([]) == ""


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 3

    def test_all_models_present(self, client):
        ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
        assert ids == {"claude-opus-4-6", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"}

    def test_model_shape(self, client):
        model = client.get("/v1/models").json()["data"][0]
        assert model["object"] == "model"
        assert model["owned_by"] == "anthropic"


# ---------------------------------------------------------------------------
# Anthropic backend: non-streaming
# ---------------------------------------------------------------------------


def _mock_response(text="Hello!", stop_reason="end_turn", input_tokens=10, output_tokens=5):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=usage)


class TestAnthropicNonStreaming:
    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
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

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_usage(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        usage = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }).json()["usage"]

        assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_kwargs_forwarded(self, mock_get_client, client):
        """Verify _build_anthropic_kwargs output reaches the SDK call."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        mock_get_client.return_value = mock_client

        client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 0.3,
            "max_tokens": 99,
        })

        kw = mock_client.messages.create.call_args[1]
        assert kw["model"] == "claude-sonnet-4-5-20250929"
        assert kw["system"] == "sys"
        assert kw["temperature"] == 0.3
        assert kw["max_tokens"] == 99

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_max_tokens_finish_reason(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_response("trunc", stop_reason="max_tokens")
        )
        mock_get_client.return_value = mock_client

        data = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }).json()

        assert data["choices"][0]["finish_reason"] == "length"

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_multi_text_blocks_concatenated(self, mock_get_client, client):
        b1 = SimpleNamespace(type="text", text="Hello ")
        b2 = SimpleNamespace(type="text", text="world!")
        b3 = SimpleNamespace(type="tool_use", id="x", name="t", input={})
        usage = SimpleNamespace(input_tokens=5, output_tokens=3)
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=[b1, b2, b3], stop_reason="end_turn", usage=usage)
        )
        mock_get_client.return_value = mock_client

        content = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }).json()["choices"][0]["message"]["content"]

        assert content == "Hello world!"


# ---------------------------------------------------------------------------
# Anthropic backend: streaming (v2 uses get_final_message for usage)
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
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

        # Final chunk should have finish_reason and usage from get_final_message()
        final_chunk = json.loads(data_lines[-2])
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert final_chunk["usage"]["prompt_tokens"] == 8
        assert final_chunk["usage"]["completion_tokens"] == 4
        assert final_chunk["usage"]["total_tokens"] == 12

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    @pytest.mark.asyncio
    async def test_streaming_stop_reason_from_final_message(self, mock_get_client, async_client):
        """Verify streaming uses get_final_message().stop_reason, not hardcoded 'stop'."""
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

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_auth_error(self, mock_get_client, client):
        import httpx

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=openai_proxy_v2.anthropic.AuthenticationError(
                message="bad key",
                response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
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

    @patch.object(openai_proxy_v2, "BACKEND", "anthropic")
    @patch.object(openai_proxy_v2, "_get_anthropic_client")
    def test_generic_exception(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
        mock_get_client.return_value = mock_client

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 500
        assert "boom" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# CLI backend
# ---------------------------------------------------------------------------


class TestCLINonStreaming:
    @patch.object(openai_proxy_v2, "BACKEND", "cli")
    @patch("openai_proxy_v2.asyncio.create_subprocess_exec")
    def test_success(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "CLI says hi"}).encode(), b"")
        )
        mock_exec.return_value = mock_proc

        data = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }).json()

        assert data["choices"][0]["message"]["content"] == "CLI says hi"

    @patch.object(openai_proxy_v2, "BACKEND", "cli")
    @patch("openai_proxy_v2.asyncio.create_subprocess_exec")
    def test_error(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fail"))
        mock_exec.return_value = mock_proc

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 502

    @patch.object(openai_proxy_v2, "BACKEND", "cli")
    @patch("openai_proxy_v2.asyncio.create_subprocess_exec")
    def test_plain_text_fallback(self, mock_exec, client):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"plain text", b""))
        mock_exec.return_value = mock_proc

        content = client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        }).json()["choices"][0]["message"]["content"]

        assert content == "plain text"

    @patch.object(openai_proxy_v2, "BACKEND", "cli")
    @patch("openai_proxy_v2.asyncio.create_subprocess_exec")
    def test_claudecode_excluded_from_env(self, mock_exec, client):
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


class TestCLIStreaming:
    @patch.object(openai_proxy_v2, "BACKEND", "cli")
    @patch("openai_proxy_v2.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_streaming(self, mock_exec, async_client):
        async def aiter_lines():
            yield json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}).encode() + b"\n"
            yield json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " there"}}).encode() + b"\n"

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
        assert data_lines[-1] == "[DONE]"

        chunks = []
        for dl in data_lines:
            if dl == "[DONE]":
                break
            c = json.loads(dl)["choices"][0]["delta"].get("content")
            if c:
                chunks.append(c)
        assert "Hi" in chunks
        assert " there" in chunks


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingletonClient:
    def test_anthropic_client_reused(self):
        """_get_anthropic_client returns the same instance on repeated calls."""
        with patch("openai_proxy_v2.anthropic.AsyncAnthropic") as MockClass:
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance

            from openai_proxy_v2 import _get_anthropic_client
            c1 = _get_anthropic_client()
            c2 = _get_anthropic_client()

            assert c1 is c2
            MockClass.assert_called_once()

    def test_cli_env_cached(self):
        """_get_cli_env computes once and caches."""
        from openai_proxy_v2 import _get_cli_env
        e1 = _get_cli_env()
        e2 = _get_cli_env()
        assert e1 is e2
