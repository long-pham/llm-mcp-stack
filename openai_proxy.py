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
import logging
import os
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("openai-proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND = os.environ.get("OPENAI_PROXY_BACKEND", "anthropic")
PORT = int(os.environ.get("OPENAI_PROXY_PORT", "8090"))
DEFAULT_MODEL = os.environ.get(
    "OPENAI_PROXY_DEFAULT_MODEL", "claude-sonnet-4-5-20250929"
)

# Optional bearer token — if set, all requests must include it.
API_KEY = os.environ.get("OPENAI_PROXY_API_KEY", "")

# CORS origins (comma-separated).  Empty = allow all.
_cors_raw = os.environ.get("OPENAI_PROXY_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()] or ["*"]

# CLI-specific settings
CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")
CLAUDE_FLAGS_RAW = os.environ.get("CLAUDE_FLAGS", "")
CLAUDE_FLAGS = shlex.split(CLAUDE_FLAGS_RAW) if CLAUDE_FLAGS_RAW else []

# Map common OpenAI model names to Claude equivalents
MODEL_MAP: dict[str, str] = {
    "gpt-4": "claude-sonnet-4-5-20250929",
    "gpt-4o": "claude-sonnet-4-5-20250929",
    "gpt-4o-mini": "claude-haiku-4-5-20251001",
    "gpt-4-turbo": "claude-sonnet-4-5-20250929",
    "gpt-3.5-turbo": "claude-haiku-4-5-20251001",
    "gpt-4.1": "claude-sonnet-4-5-20250929",
    "gpt-4.1-mini": "claude-haiku-4-5-20251001",
    "gpt-4.1-nano": "claude-haiku-4-5-20251001",
    "o1": "claude-opus-4-6",
    "o1-preview": "claude-opus-4-6",
    "o3": "claude-opus-4-6",
    "o3-mini": "claude-sonnet-4-5-20250929",
    "o4-mini": "claude-sonnet-4-5-20250929",
}

AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
]

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
    ]
    + [
        {
            "id": alias,
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
            "description": f"Maps to {target}",
        }
        for alias, target in MODEL_MAP.items()
    ],
}

STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _anthropic_client
    if _anthropic_client is not None:
        await _anthropic_client.close()
        _anthropic_client = None


app = FastAPI(title="OpenAI-Compatible Claude Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def check_auth(request: Request, call_next):
    if API_KEY and request.url.path not in ("/health",):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            return _openai_error(
                "Invalid or missing API key",
                status=401,
                err_type="authentication_error",
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model(model: str | None) -> str:
    """Resolve an OpenAI model name to a Claude model id.

    Known aliases are mapped; unknown names are passed through as-is
    so the upstream API can decide validity.
    """
    if not model:
        return DEFAULT_MODEL
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    return model


def _openai_error(
    message: str,
    status: int = 400,
    err_type: str = "invalid_request_error",
) -> JSONResponse:
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


def _map_stop_reason(reason: str | None) -> str:
    return STOP_REASON_MAP.get(reason, "stop")


def _translate_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system prompt and convert messages to Anthropic format."""
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
            system_parts.append(
                content if isinstance(content, str) else json.dumps(content)
            )
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


def _build_anthropic_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    """Build Anthropic API kwargs from an OpenAI-format request body."""
    model = _resolve_model(body.get("model"))
    system_prompt, messages = _translate_messages(body.get("messages", []))

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = 4096

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    temperature = body.get("temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature

    top_p = body.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p

    stop = body.get("stop")
    if stop is not None:
        # OpenAI accepts a string or list; Anthropic expects a list.
        if isinstance(stop, str):
            stop = [stop]
        kwargs["stop_sequences"] = stop

    return kwargs


def _make_chunk(
    comp_id: str,
    created: int,
    model: str,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-compatible streaming chunk."""
    delta: dict[str, str] = {}
    if content:
        delta["content"] = content

    chunk: dict[str, Any] = {
        "id": comp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


# ---------------------------------------------------------------------------
# Anthropic SDK Backend
# ---------------------------------------------------------------------------


async def _anthropic_chat(body: dict[str, Any]) -> JSONResponse:
    """Non-streaming chat completion via Anthropic API."""
    kwargs = _build_anthropic_kwargs(body)
    model = kwargs["model"]
    client = _get_anthropic_client()

    response = await client.messages.create(**kwargs)

    text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return JSONResponse(
        content={
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
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
        }
    )


async def _anthropic_chat_stream(body: dict[str, Any]):
    """Streaming chat completion via Anthropic API."""
    kwargs = _build_anthropic_kwargs(body)
    model = kwargs["model"]
    comp_id = _completion_id()
    created = _timestamp()
    client = _get_anthropic_client()

    async def event_generator():
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield {
                    "data": json.dumps(
                        _make_chunk(comp_id, created, model, content=text)
                    )
                }

            final = await stream.get_final_message()

        usage = {
            "prompt_tokens": final.usage.input_tokens,
            "completion_tokens": final.usage.output_tokens,
            "total_tokens": final.usage.input_tokens
            + final.usage.output_tokens,
        }
        yield {
            "data": json.dumps(
                _make_chunk(
                    comp_id,
                    created,
                    model,
                    finish_reason=_map_stop_reason(final.stop_reason),
                    usage=usage,
                )
            )
        }
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# CLI Backend
# ---------------------------------------------------------------------------


def _get_cli_env() -> dict[str, str]:
    """Build a clean env for CLI subprocesses (computed fresh each call)."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _messages_to_prompt_string(messages: list[dict[str, Any]]) -> str:
    """Flatten OpenAI-format messages into a single prompt string for the CLI."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if role == "system":
            parts.append(f"[System]: {content}")
        elif role == "assistant":
            parts.append(f"[Assistant]: {content}")
        else:
            parts.append(f"[User]: {content}")
    return "\n\n".join(parts)


def _build_cli_cmd(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
    stream: bool = False,
    session_id: str | None = None,
) -> list[str]:
    """Build the claude CLI command list."""
    cmd = [CLAUDE_CLI, "-p", prompt, "--model", model]

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])

    if stream:
        cmd.extend(["--output-format", "stream-json", "--verbose"])
    else:
        cmd.extend(["--output-format", "json"])

    if session_id:
        cmd.extend(["--session-id", session_id])

    if CLAUDE_FLAGS:
        cmd.extend(CLAUDE_FLAGS)

    return cmd


def _extract_cli_text(event: dict[str, Any]) -> str | None:
    """Extract text content from a CLI stream event.

    Only handles content_block_delta to avoid double-emitting text that
    also appears in the final 'result' event.
    """
    event_type = event.get("type")
    if event_type == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text", "")
    elif event_type == "assistant" and "message" in event:
        msg = event["message"]
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    return None


async def _cli_chat(
    body: dict[str, Any], session_id: str | None = None
) -> JSONResponse:
    """Non-streaming chat completion via Claude Code CLI."""
    model = _resolve_model(body.get("model"))
    messages = body.get("messages", [])
    prompt = _messages_to_prompt_string(messages)
    system_prompt, _ = _translate_messages(messages)

    max_tokens = body.get("max_tokens")

    cmd = _build_cli_cmd(
        prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        session_id=session_id,
    )

    log.info("CLI exec: %s", " ".join(shlex.quote(c) for c in cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_get_cli_env(),
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        log.error("Claude CLI not found: %s", CLAUDE_CLI)
        return _openai_error(
            f"Claude CLI not found: {CLAUDE_CLI}",
            status=500,
            err_type="configuration_error",
        )

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        log.error("CLI error (rc=%d): %s", proc.returncode, err_msg)
        return _openai_error(
            f"CLI error (exit {proc.returncode}): {err_msg}",
            status=502,
            err_type="upstream_error",
        )

    output = stdout.decode()
    try:
        result = json.loads(output)
        text = result.get("result", output)
        resp_session_id = result.get("session_id", session_id)
        usage_input = result.get("num_input_tokens", 0)
        usage_output = result.get("num_output_tokens", 0)
    except json.JSONDecodeError:
        text = output
        resp_session_id = session_id
        usage_input = 0
        usage_output = 0

    comp_id = (
        f"chatcmpl-{resp_session_id}"
        if resp_session_id
        else _completion_id()
    )

    headers = {}
    if resp_session_id:
        headers["x-session-id"] = resp_session_id

    return JSONResponse(
        content={
            "id": comp_id,
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
                "prompt_tokens": usage_input,
                "completion_tokens": usage_output,
                "total_tokens": usage_input + usage_output,
            },
        },
        headers=headers,
    )


async def _cli_chat_stream(
    body: dict[str, Any], session_id: str | None = None
):
    """Streaming chat completion via Claude Code CLI."""
    model = _resolve_model(body.get("model"))
    messages = body.get("messages", [])
    comp_id = _completion_id()
    created = _timestamp()
    prompt = _messages_to_prompt_string(messages)
    system_prompt, _ = _translate_messages(messages)

    max_tokens = body.get("max_tokens")

    cmd = _build_cli_cmd(
        prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        stream=True,
        session_id=session_id,
    )

    log.info("CLI stream: %s", " ".join(shlex.quote(c) for c in cmd))

    async def event_generator():
        nonlocal comp_id

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_get_cli_env(),
            )
        except FileNotFoundError:
            log.error("Claude CLI not found: %s", CLAUDE_CLI)
            yield {
                "data": json.dumps(
                    _make_chunk(
                        comp_id,
                        created,
                        model,
                        content=f"[Error: Claude CLI not found: {CLAUDE_CLI}]",
                    )
                )
            }
            yield {"data": "[DONE]"}
            return

        sent_any = False
        async for line in proc.stdout:
            line = line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Pick up session_id from system event
            if event.get("type") == "system" and event.get("session_id"):
                comp_id = f"chatcmpl-{event['session_id']}"

            text = _extract_cli_text(event)
            if text:
                yield {
                    "data": json.dumps(
                        _make_chunk(comp_id, created, model, content=text)
                    )
                }
                sent_any = True

        await proc.wait()

        if proc.returncode != 0:
            log.error("CLI stream error (rc=%d)", proc.returncode)
            if not sent_any:
                yield {
                    "data": json.dumps(
                        _make_chunk(
                            comp_id,
                            created,
                            model,
                            content="[Error: Claude CLI exited with an error]",
                        )
                    )
                }

        yield {
            "data": json.dumps(
                _make_chunk(
                    comp_id, created, model, finish_reason="stop"
                )
            )
        }
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


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
async def chat_completions(
    request: Request,
    x_session_id: str | None = Header(None),
):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body")

    stream = body.get("stream", False)

    try:
        if BACKEND == "cli":
            if stream:
                return await _cli_chat_stream(body, session_id=x_session_id)
            return await _cli_chat(body, session_id=x_session_id)
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
    except Exception:
        log.exception("Unexpected error in chat completion")
        return _openai_error(
            "Internal server error", status=500, err_type="server_error"
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(
        "Starting OpenAI-compatible Claude proxy (backend=%s) on port %d",
        BACKEND,
        PORT,
    )
    log.info("Claude CLI: %s", CLAUDE_CLI)
    log.info("Default model: %s", DEFAULT_MODEL)
    log.info("Auth: %s", "enabled" if API_KEY else "disabled")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
