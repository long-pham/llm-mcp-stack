"""OpenAI API-compatible proxy that relays requests to Claude.

Supports two backends:
  - "anthropic": Uses the Anthropic Python SDK (requires ANTHROPIC_API_KEY)
  - "cli": Invokes the Claude Code CLI via subprocess

Usage:
  ANTHROPIC_API_KEY=sk-... uv run python openai_proxy.py
  OPENAI_PROXY_BACKEND=cli uv run python openai_proxy.py
"""

from __future__ import annotations

import asyncio
import json
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

BACKEND = os.environ.get("OPENAI_PROXY_BACKEND", "anthropic")
PORT = int(os.environ.get("OPENAI_PROXY_PORT", "8090"))
DEFAULT_MODEL = os.environ.get(
    "OPENAI_PROXY_DEFAULT_MODEL", "claude-sonnet-4-5-20250929"
)

# Map common OpenAI model names to Claude equivalents
MODEL_MAP: dict[str, str] = {
    "gpt-4": "claude-sonnet-4-5-20250929",
    "gpt-4o": "claude-sonnet-4-5-20250929",
    "gpt-4o-mini": "claude-haiku-4-5-20251001",
    "gpt-4-turbo": "claude-sonnet-4-5-20250929",
    "gpt-3.5-turbo": "claude-haiku-4-5-20251001",
    "o1": "claude-opus-4-6",
    "o1-preview": "claude-opus-4-6",
}

AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
]

# Pre-built response for /v1/models (static data, no need to rebuild per request)
_MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": m["id"],
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        }
        for m in AVAILABLE_MODELS
    ],
}

# Pre-compute CLI environment (exclude CLAUDECODE to avoid nested-session errors)
_CLI_ENV: dict[str, str] | None = None

STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}

app = FastAPI(title="OpenAI-Compatible Claude Proxy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model(model: str | None) -> str:
    """Resolve an OpenAI model name to a Claude model id."""
    if not model:
        return DEFAULT_MODEL
    if model.startswith("claude-"):
        return model
    return MODEL_MAP.get(model, DEFAULT_MODEL)


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


def _timestamp() -> int:
    return int(time.time())


def _translate_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system prompt and convert messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
            continue

        # Map "assistant" stays "assistant", everything else becomes "user"
        anthropic_role = "assistant" if role == "assistant" else "user"

        # Merge consecutive same-role messages
        if converted and converted[-1]["role"] == anthropic_role:
            prev = converted[-1]["content"]
            if isinstance(prev, str) and isinstance(content, str):
                converted[-1]["content"] = prev + "\n" + content
            else:
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
# Anthropic API Backend
# ---------------------------------------------------------------------------


_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


async def _anthropic_chat(body: dict[str, Any]) -> JSONResponse:
    """Non-streaming chat completion via Anthropic API."""
    model = _resolve_model(body.get("model"))
    system_prompt, messages = _translate_messages(body.get("messages", []))
    max_tokens = body.get("max_tokens") or 4096
    temperature = body.get("temperature")

    client = _get_anthropic_client()
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

    # Build OpenAI-format response
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    return JSONResponse(content={
        "id": _completion_id(),
        "object": "chat.completion",
        "created": _timestamp(),
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


def _map_stop_reason(reason: str | None) -> str:
    return STOP_REASON_MAP.get(reason, "stop")


async def _anthropic_chat_stream(body: dict[str, Any]):
    """Streaming chat completion via Anthropic API."""
    model = _resolve_model(body.get("model"))
    system_prompt, messages = _translate_messages(body.get("messages", []))
    max_tokens = body.get("max_tokens") or 4096
    temperature = body.get("temperature")
    comp_id = _completion_id()
    created = _timestamp()

    client = _get_anthropic_client()
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
# CLI Backend
# ---------------------------------------------------------------------------


def _get_cli_env() -> dict[str, str]:
    global _CLI_ENV
    if _CLI_ENV is None:
        _CLI_ENV = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    return _CLI_ENV


async def _cli_chat(body: dict[str, Any]) -> JSONResponse:
    """Non-streaming chat completion via Claude Code CLI."""
    model = _resolve_model(body.get("model"))
    messages = body.get("messages", [])

    prompt = _messages_to_prompt_string(messages)

    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_get_cli_env(),
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return _openai_error(
            f"CLI error (exit {proc.returncode}): {stderr.decode(errors='replace')}",
            status=502,
            err_type="upstream_error",
        )

    output = stdout.decode()
    try:
        result = json.loads(output)
        text = result.get("result", output)
    except json.JSONDecodeError:
        text = output

    return JSONResponse(content={
        "id": _completion_id(),
        "object": "chat.completion",
        "created": _timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


async def _cli_chat_stream(body: dict[str, Any]):
    """Streaming chat completion via Claude Code CLI."""
    model = _resolve_model(body.get("model"))
    messages = body.get("messages", [])
    comp_id = _completion_id()
    created = _timestamp()

    prompt = _messages_to_prompt_string(messages)

    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_get_cli_env(),
    )

    async def event_generator():
        # Pre-build envelope for fast per-token serialization
        prefix = json.dumps({
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        })
        marker = '"content": ""'
        split = prefix.index(marker) + len('"content": "')
        pre, suf = prefix[:split], prefix[split:]

        async for line in proc.stdout:
            line = line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract text from various CLI event types
            text = _extract_cli_text(event)

            if text:
                escaped = json.dumps(text)[1:-1]
                yield {"data": pre + escaped + suf}

        await proc.wait()

        yield {"data": json.dumps({
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


def _extract_cli_text(event: dict[str, Any]) -> str | None:
    """Extract text content from a CLI stream event."""
    event_type = event.get("type")
    if event_type == "assistant" and "message" in event:
        msg = event["message"]
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    elif event_type == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text", "")
    elif event_type == "result":
        return event.get("result", "")
    return None


def _messages_to_prompt_string(messages: list[dict[str, Any]]) -> str:
    """Flatten OpenAI-format messages into a single prompt string for the CLI."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if role == "system":
            parts.append(f"[System]: {content}")
        elif role == "assistant":
            parts.append(f"[Assistant]: {content}")
        else:
            parts.append(f"[User]: {content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND}


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
        if BACKEND == "cli":
            if stream:
                return await _cli_chat_stream(body)
            return await _cli_chat(body)
        else:
            if stream:
                return await _anthropic_chat_stream(body)
            return await _anthropic_chat(body)
    except anthropic.AuthenticationError as e:
        return _openai_error(str(e), status=401, err_type="authentication_error")
    except anthropic.RateLimitError as e:
        return _openai_error(str(e), status=429, err_type="rate_limit_error")
    except anthropic.APIError as e:
        return _openai_error(str(e), status=502, err_type="upstream_error")
    except Exception as e:
        return _openai_error(f"Internal error: {e}", status=500, err_type="server_error")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting OpenAI-compatible Claude proxy (backend={BACKEND}) on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
