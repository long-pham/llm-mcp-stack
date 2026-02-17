"""Tests for the Anthropic SDK-based OpenAI proxy."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
import httpx
from httpx import ASGITransport

from proxy_sdk import (
    app,
    _resolve_model,
    _translate_messages,
    _map_stop_reason,
    DEFAULT_MODEL,
)


# ---------------------------------------------------------------------------
# Unit tests: _resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_maps_gpt4_to_sonnet(self):
        assert _resolve_model("gpt-4") == "claude-sonnet-4-5-20250929"

    def test_maps_gpt4o_to_sonnet(self):
        assert _resolve_model("gpt-4o") == "claude-sonnet-4-5-20250929"

    def test_maps_gpt35_to_haiku(self):
        assert _resolve_model("gpt-3.5-turbo") == "claude-haiku-4-5-20251001"

    def test_maps_gpt4o_mini_to_haiku(self):
        assert _resolve_model("gpt-4o-mini") == "claude-haiku-4-5-20251001"

    def test_maps_o3_to_opus(self):
        assert _resolve_model("o3") == "claude-opus-4-6"

    def test_passes_through_claude_model(self):
        assert _resolve_model("claude-opus-4-6") == "claude-opus-4-6"

    def test_passes_through_unknown_model(self):
        assert _resolve_model("some-custom-model") == "some-custom-model"

    def test_none_returns_default(self):
        assert _resolve_model(None) == DEFAULT_MODEL

    def test_empty_string_returns_default(self):
        assert _resolve_model("") == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Unit tests: _translate_messages
# ---------------------------------------------------------------------------


class TestTranslateMessages:
    def test_single_user_message(self):
        system, msgs = _translate_messages([{"role": "user", "content": "Hello"}])
        assert system is None
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_system_message_extracted(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "You are helpful"
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_multiple_system_messages(self):
        system, msgs = _translate_messages([
            {"role": "system", "content": "Be concise"},
            {"role": "system", "content": "Use formal language"},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "Be concise\n\nUse formal language"

    def test_consecutive_same_role_merged(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello\nWorld"

    def test_assistant_first_gets_user_prefix(self):
        system, msgs = _translate_messages([
            {"role": "assistant", "content": "I already said something"},
        ])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "(continued conversation)"
        assert msgs[1]["role"] == "assistant"

    def test_empty_messages_gets_fallback(self):
        system, msgs = _translate_messages([])
        assert system is None
        assert msgs == [{"role": "user", "content": "(empty)"}]

    def test_content_as_list_of_parts(self):
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

    def test_multi_turn_conversation(self):
        system, msgs = _translate_messages([
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ])
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"


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
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "sdk"


@pytest.mark.asyncio
async def test_models_endpoint(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "claude-opus-4-6" in model_ids
    assert "claude-sonnet-4-5-20250929" in model_ids
    assert "claude-haiku-4-5-20251001" in model_ids
    # OpenAI aliases present
    assert "gpt-4" in model_ids
    assert "gpt-3.5-turbo" in model_ids


@pytest.mark.asyncio
async def test_models_all_have_required_fields(client):
    resp = await client.get("/v1/models")
    data = resp.json()
    for model in data["data"]:
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_response(text: str = "Hello!", stop_reason: str = "end_turn",
                        input_tokens: int = 10, output_tokens: int = 5):
    """Build a mock Anthropic Messages response."""
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=usage)


class MockTextStream:
    """Async iterator that yields text chunks."""
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class MockStreamManager:
    """Async context manager mimicking client.messages.stream()."""
    def __init__(self, chunks: list[str]):
        self.text_stream = MockTextStream(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Integration tests: non-streaming chat completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_non_streaming(client):
    mock_resp = _make_mock_response("How can I help?", input_tokens=12, output_tokens=8)
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    with patch("proxy_sdk._get_client", return_value=mock_client):
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
    assert data["choices"][0]["message"]["content"] == "How can I help?"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 12
    assert data["usage"]["completion_tokens"] == 8
    assert data["usage"]["total_tokens"] == 20
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "claude-sonnet-4-5-20250929"


@pytest.mark.asyncio
async def test_chat_non_streaming_max_tokens_stop(client):
    mock_resp = _make_mock_response("truncated", stop_reason="max_tokens")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Write a long essay"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_chat_passes_temperature(client):
    mock_resp = _make_mock_response("ok")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.7,
            },
        )

    assert resp.status_code == 200
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_chat_passes_system_prompt(client):
    mock_resp = _make_mock_response("Ahoy!")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    with patch("proxy_sdk._get_client", return_value=mock_client):
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
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a pirate"


@pytest.mark.asyncio
async def test_chat_passes_max_tokens(client):
    mock_resp = _make_mock_response("ok")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 256,
            },
        )

    assert resp.status_code == 200
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# Integration tests: streaming chat completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_streaming(client):
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(
        return_value=MockStreamManager(["Hello", " world", "!"])
    )

    with patch("proxy_sdk._get_client", return_value=mock_client):
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

    # Parse SSE events
    events = []
    for line in resp.text.split("\n"):
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
    assert len(content_chunks) == 3

    # Concatenated content should match
    full = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
    assert full == "Hello world!"

    # Last event should be [DONE]
    assert events[-1] == "[DONE]"

    # Second-to-last should be finish chunk
    finish_events = [
        e for e in events
        if isinstance(e, dict) and e.get("choices", [{}])[0].get("finish_reason") == "stop"
    ]
    assert len(finish_events) == 1


# ---------------------------------------------------------------------------
# Integration tests: error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_error_returns_401(client):
    mock_client = MagicMock()
    error = anthropic.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API key"}},
    )
    mock_client.messages.create = AsyncMock(side_effect=error)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_rate_limit_error_returns_429(client):
    mock_client = MagicMock()
    error = anthropic.RateLimitError(
        message="Rate limited",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limited"}},
    )
    mock_client.messages.create = AsyncMock(side_effect=error)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_api_error_returns_502(client):
    mock_client = MagicMock()
    error = anthropic.APIError(
        message="Server error",
        request=MagicMock(),
        body={"error": {"message": "Server error"}},
    )
    mock_client.messages.create = AsyncMock(side_effect=error)

    with patch("proxy_sdk._get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


@pytest.mark.asyncio
async def test_invalid_json_body(client):
    resp = await client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
