"""
OpenAI-compatible API proxy that calls the Anthropic API directly via the SDK.

Much faster than the CLI-based proxy (proxy.py) because it avoids spawning
a subprocess per request and reuses HTTP connections.

Exposes /v1/chat/completions and /v1/models so any OpenAI-compatible client
(Continue, Cursor, Open WebUI, etc.) can use Claude as a backend.

Usage:
  ANTHROPIC_API_KEY=sk-... python proxy_sdk.py
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8082"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("openai-proxy-sdk")

# ---------------------------------------------------------------------------
# Model mapping
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "gpt-4": "claude-sonnet-4-5-20250929",
    "gpt-4o": "claude-sonnet-4-5-20250929",
    "gpt-4o-mini": "claude-haiku-4-5-20251001",
    "gpt-4-turbo": "claude-sonnet-4-5-20250929",
    "gpt-3.5-turbo": "claude-haiku-4-5-20251001",
    "gpt-4.1": "claude-sonnet-4-5-20250929",
    "gpt-4.1-mini": "claude-haiku-4-5-20251001",
    "gpt-4.1-nano": "claude-haiku-4-5-20251001",
    "o3": "claude-opus-4-6",
    "o3-mini": "claude-sonnet-4-5-20250929",
    "o4-mini": "claude-sonnet-4-5-20250929",
}

AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
]

STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}

# Pre-built models response (static data, no need to rebuild per request)
_MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": m["id"], "object": "model", "created": 1700000000, "owned_by": "anthropic"}
        for m in AVAILABLE_MODELS
    ] + [
        {"id": alias, "object": "model", "created": 1700000000, "owned_by": "anthropic", "description": f"Maps to {target}"}
        for alias, target in MODEL_MAP.items()
    ],
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenAI-to-Claude SDK Proxy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Lazy-init singleton Anthropic client (reuses HTTP connections)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _resolve_model(model: str | None) -> str:
    """Map an OpenAI model name to a Claude model, or pass through directly."""
    if not model:
        return DEFAULT_MODEL
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    return model


def _openai_error(message: str, status: int = 400, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": None,
            }
        },
    )


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _map_stop_reason(reason: str | None) -> str:
    return STOP_REASON_MAP.get(reason, "stop")


def _translate_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system prompt and convert messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle content that is a list of content parts
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
            continue

        # Map "assistant" stays "assistant", everything else becomes "user"
        anthropic_role = "assistant" if role == "assistant" else "user"

        # Merge consecutive same-role messages
        if converted and converted[-1]["role"] == anthropic_role:
            prev = converted[-1]["content"]
            converted[-1]["content"] = str(prev) + "\n" + str(content)
        else:
            converted.append({"role": anthropic_role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # Ensure the first message is from "user"
    if converted and converted[0]["role"] != "user":
        converted.insert(0, {"role": "user", "content": "(continued conversation)"})

    # Ensure we have at least one message
    if not converted:
        converted.append({"role": "user", "content": "(empty)"})

    return system_prompt, converted


# ---------------------------------------------------------------------------
# Non-streaming completion
# ---------------------------------------------------------------------------


async def _chat(body: dict[str, Any]) -> JSONResponse:
    """Non-streaming chat completion via Anthropic API."""
    model = _resolve_model(body.get("model"))
    system_prompt, messages = _translate_messages(body.get("messages", []))
    max_tokens = body.get("max_tokens") or 4096
    temperature = body.get("temperature")

    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await client.messages.create(**kwargs)

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    return JSONResponse(content={
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _map_stop_reason(response.stop_reason),
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    })


# ---------------------------------------------------------------------------
# Streaming completion
# ---------------------------------------------------------------------------


async def _chat_stream(body: dict[str, Any]):
    """Streaming chat completion via Anthropic API."""
    model = _resolve_model(body.get("model"))
    system_prompt, messages = _translate_messages(body.get("messages", []))
    max_tokens = body.get("max_tokens") or 4096
    temperature = body.get("temperature")
    comp_id = _completion_id()
    created = int(time.time())

    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature is not None:
        kwargs["temperature"] = temperature

    async def event_generator():
        # Pre-build envelope prefix/suffix to avoid repeated dict construction + serialization
        prefix = json.dumps({
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        })
        # Find the split point around the empty content value
        marker = '"content": ""'
        split = prefix.index(marker) + len('"content": "')
        pre, suf = prefix[:split], prefix[split:]

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                # Escape text for JSON string embedding
                escaped = json.dumps(text)[1:-1]
                yield {"data": pre + escaped + suf}

        yield {"data": json.dumps({
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "sdk"}


@app.get("/v1/models")
async def list_models():
    return _MODELS_RESPONSE


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body")

    stream = body.get("stream", False)

    try:
        if stream:
            return await _chat_stream(body)
        return await _chat(body)
    except anthropic.AuthenticationError as e:
        return _openai_error(str(e), status=401, err_type="authentication_error")
    except anthropic.RateLimitError as e:
        return _openai_error(str(e), status=429, err_type="rate_limit_error")
    except anthropic.APIError as e:
        return _openai_error(str(e), status=502, err_type="upstream_error")
    except Exception as e:
        log.exception("Unexpected error in chat completion")
        return _openai_error(f"Internal error: {e}", status=500, err_type="server_error")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting OpenAI-compatible Claude SDK proxy on port %d", PORT)
    log.info("Default model: %s", DEFAULT_MODEL)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
