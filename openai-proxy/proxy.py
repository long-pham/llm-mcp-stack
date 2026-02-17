"""
OpenAI-compatible API proxy that relays requests to the Claude Code CLI.

Exposes /v1/chat/completions and /v1/models so any OpenAI-compatible client
(Continue, Cursor, Open WebUI, etc.) can use Claude Code as a backend.
"""

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8082"))
CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
CLAUDE_FLAGS_RAW = os.environ.get("CLAUDE_FLAGS", "")
CLAUDE_FLAGS = shlex.split(CLAUDE_FLAGS_RAW) if CLAUDE_FLAGS_RAW else []

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("openai-proxy")

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

app = FastAPI(title="OpenAI-to-Claude Code Proxy")


def resolve_model(client_model: Optional[str]) -> str:
    """Map an OpenAI model name to a Claude model, or pass through directly."""
    if not client_model:
        return DEFAULT_MODEL
    if client_model in MODEL_MAP:
        return MODEL_MAP[client_model]
    # Pass through Claude model names directly
    return client_model


def build_prompt(messages: list[dict]) -> tuple[str, Optional[str]]:
    """
    Convert OpenAI-style messages into a single prompt string for ``claude -p``.

    Returns (prompt, system_prompt | None).
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

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
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(f"Human: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    prompt = "\n\n".join(conversation_parts)
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return prompt, system_prompt


def build_claude_cmd(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    session_id: Optional[str] = None,
) -> list[str]:
    """Build the claude CLI command list."""
    cmd = [CLAUDE_CLI, "-p", prompt]

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    cmd.extend(["--model", model])

    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])

    if stream:
        cmd.extend(["--output-format", "stream-json", "--verbose"])
    else:
        cmd.extend(["--output-format", "json"])

    if session_id:
        cmd.extend(["--session-id", session_id])

    # Append any extra flags from env
    if CLAUDE_FLAGS:
        cmd.extend(CLAUDE_FLAGS)

    return cmd


def make_completion_response(
    content: str,
    model: str,
    session_id: Optional[str] = None,
    usage: Optional[dict] = None,
) -> dict:
    """Build an OpenAI-compatible chat completion response."""
    comp_id = f"chatcmpl-{session_id or uuid.uuid4().hex[:24]}"
    return {
        "id": comp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def make_chunk(
    content: str,
    model: str,
    chunk_id: str,
    finish_reason: Optional[str] = None,
    include_role: bool = False,
) -> dict:
    """Build a single OpenAI-compatible streaming chunk."""
    delta: dict = {}
    if include_role:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return _MODELS_RESPONSE


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_session_id: Optional[str] = Header(None),
):
    body = await request.json()

    messages = body.get("messages", [])
    client_model = body.get("model")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens")

    model = resolve_model(client_model)
    prompt, system_prompt = build_prompt(messages)

    if not prompt:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user messages provided", "type": "invalid_request_error"}},
        )

    session_id = x_session_id

    cmd = build_claude_cmd(
        prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        stream=stream,
        session_id=session_id,
    )

    log.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))

    if stream:
        return EventSourceResponse(
            stream_response(cmd, model, session_id),
            media_type="text/event-stream",
        )
    else:
        return await non_streaming_response(cmd, model, session_id)


async def non_streaming_response(
    cmd: list[str], model: str, session_id: Optional[str]
) -> JSONResponse:
    """Run claude CLI and return a single OpenAI-compatible response."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            log.error("claude CLI error (rc=%d): %s", proc.returncode, err_msg)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Claude CLI error: {err_msg}",
                        "type": "upstream_error",
                    }
                },
            )

        raw = stdout.decode(errors="replace").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # If not JSON, treat raw output as the response text
            return JSONResponse(
                content=make_completion_response(raw, model, session_id)
            )

        content = result.get("result", "")
        resp_session_id = result.get("session_id", session_id)

        usage_input = result.get("num_input_tokens", 0)
        usage_output = result.get("num_output_tokens", 0)
        usage = {
            "prompt_tokens": usage_input,
            "completion_tokens": usage_output,
            "total_tokens": usage_input + usage_output,
        }

        resp = make_completion_response(content, model, resp_session_id, usage)

        headers = {}
        if resp_session_id:
            headers["x-session-id"] = resp_session_id

        return JSONResponse(content=resp, headers=headers)

    except FileNotFoundError:
        log.error("claude CLI not found at: %s", CLAUDE_CLI)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Claude CLI not found: {CLAUDE_CLI}",
                    "type": "configuration_error",
                }
            },
        )
    except Exception as exc:
        log.exception("Unexpected error in non-streaming response")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "internal_error",
                }
            },
        )


async def stream_response(cmd: list[str], model: str, session_id: Optional[str]):
    """
    Async generator that yields OpenAI SSE chunks by reading NDJSON
    lines from the claude CLI's stream-json output.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    sent_role = False

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Skipping non-JSON line: %s", line[:120])
                continue

            event_type = event.get("type", "")

            # Extract session_id from initial message
            if event_type == "system" and event.get("session_id"):
                session_id = event["session_id"]
                chunk_id = f"chatcmpl-{session_id}"

            # Content blocks with text deltas
            if event_type == "assistant":
                text = event.get("message", "")
                if text:
                    chunk = make_chunk(
                        text, model, chunk_id, include_role=not sent_role
                    )
                    sent_role = True
                    yield {"data": json.dumps(chunk)}

            # content_block_delta events (stream-json verbose format)
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text", "")
                if text:
                    chunk = make_chunk(
                        text, model, chunk_id, include_role=not sent_role
                    )
                    sent_role = True
                    yield {"data": json.dumps(chunk)}

            # Result message at the end
            if event_type == "result":
                text = event.get("result", "")
                if text and not sent_role:
                    # Fallback: send entire result as one chunk
                    chunk = make_chunk(
                        text, model, chunk_id, include_role=True
                    )
                    yield {"data": json.dumps(chunk)}
                    sent_role = True

        await proc.wait()

        if proc.returncode != 0:
            log.error("claude CLI stream error (rc=%d)", proc.returncode)
            if not sent_role:
                error_chunk = make_chunk(
                    "[Error: Claude CLI exited with an error]",
                    model,
                    chunk_id,
                    include_role=True,
                )
                yield {"data": json.dumps(error_chunk)}

        # Send finish chunk and DONE
        finish_chunk = make_chunk("", model, chunk_id, finish_reason="stop")
        yield {"data": json.dumps(finish_chunk)}
        yield {"data": "[DONE]"}

    except FileNotFoundError:
        log.error("claude CLI not found at: %s", CLAUDE_CLI)
        error_chunk = make_chunk(
            f"[Error: Claude CLI not found: {CLAUDE_CLI}]",
            model,
            chunk_id,
            include_role=True,
        )
        yield {"data": json.dumps(error_chunk)}
        yield {"data": "[DONE]"}
    except Exception as exc:
        log.exception("Unexpected error in streaming response")
        error_chunk = make_chunk(
            f"[Error: {exc}]",
            model,
            chunk_id,
            include_role=True,
        )
        yield {"data": json.dumps(error_chunk)}
        yield {"data": "[DONE]"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting OpenAI-to-Claude proxy on port %d", PORT)
    log.info("Claude CLI: %s", CLAUDE_CLI)
    log.info("Default model: %s", DEFAULT_MODEL)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
