# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Docker Compose stack providing MCP (Model Context Protocol) services with an OpenAI-compatible API proxy for Claude. The stack runs SearXNG (web search), Crawl4AI (web crawling), and MCPHub (MCP gateway/aggregator), plus a FastAPI proxy that translates OpenAI API calls to Anthropic's Claude API.

## Commands

### Package Management
Always use `uv` (never pip/pipx):
```bash
uv sync                    # Install dependencies
uv sync --group dev        # Install with dev dependencies
```

### Running the Proxy
```bash
ANTHROPIC_API_KEY=sk-... uv run llm-mcp-proxy          # Anthropic SDK backend
OPENAI_PROXY_BACKEND=cli uv run llm-mcp-proxy           # Claude Code CLI backend
```

### Tests
```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_openai_proxy.py   # Run proxy unit tests
uv run pytest -k "TestResolveModel"        # Run a specific test class
uv run pytest -k "test_basic_completion"   # Run a single test
```

### Docker Services
```bash
./start.sh                  # First-time setup + start all services (auto-generates secrets)
docker compose up -d        # Start services (requires .env)
docker compose ps           # Check service status
docker compose logs -f      # Follow logs
```

## Architecture

### OpenAI Proxy (`src/llm_mcp_stack/openai_proxy.py`)
Single-file FastAPI application that implements OpenAI's `/v1/chat/completions` and `/v1/models` endpoints, translating requests to Claude. Two backends:
- **anthropic**: Uses the Anthropic Python SDK directly (default)
- **cli**: Shells out to the `claude` CLI subprocess

Key internals:
- `_translate_messages()` converts OpenAI message format to Anthropic's (extracts system prompts, merges consecutive same-role messages, ensures user-first ordering)
- `MODEL_MAP` maps OpenAI model names (gpt-4, o1, etc.) to Claude equivalents
- Streaming uses SSE via `sse-starlette`; both backends support streaming and non-streaming
- Auth middleware checks `OPENAI_PROXY_API_KEY` bearer token when set

### Docker Services (`docker-compose.yml`)
All services share the `mcp-network` bridge network:
- **searxng** (port 38080): Privacy-respecting metasearch engine
- **searxng-mcp** (port 38081): MCP server wrapping SearXNG
- **crawl4ai** (port 11235): Web crawling with headless Chrome (2GB shm, 4GB memory limit)
- **mcphub** (port 3000): MCP aggregator/gateway with file-based config (`mcphub/mcp_settings.json`)

### Test Structure
- `tests/test_openai_proxy.py` — Unit tests for the proxy (mocked Anthropic client and CLI subprocess, no external calls needed)
- `tests/mcp/` — Integration tests for MCP services (require running Docker containers)

## Configuration
All configuration via environment variables in `.env` (see `.env.example`). The `start.sh` script auto-generates SEARXNG_SECRET_KEY on first run.
