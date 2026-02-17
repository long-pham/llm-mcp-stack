# GEMINI.md

This file provides instructional context for Gemini CLI when working in the `llm-mcp-stack` repository.

## Project Overview

The **llm-mcp-stack** is a Docker Compose-based environment designed to provide a suite of MCP (Model Context Protocol) services alongside an OpenAI-compatible API proxy for Claude. This allows tools and applications expecting an OpenAI API to leverage Claude's capabilities while having access to various MCP tools for web search, crawling, and more.

### Key Components:
- **OpenAI Proxy (`src/llm_mcp_stack/openai_proxy.py`)**: A FastAPI application that translates OpenAI's `/v1/chat/completions` and `/v1/models` endpoints to Anthropic's Claude API.
    - **Backends**: Supports `anthropic` (SDK-based) and `cli` (shells out to `claude` CLI).
- **SearXNG**: Privacy-respecting metasearch engine.
- **SearXNG MCP Server**: Connects SearXNG to the MCP ecosystem.
- **Crawl4AI**: High-performance web crawling and scraping service.
- **MCPHub**: An MCP aggregator and gateway with file-based config, used to manage multiple MCP servers.

### Main Technologies:
- **Python 3.12+** (FastAPI, Anthropic SDK, Uvicorn)
- **uv** for dependency management.
- **Docker & Docker Compose** for service orchestration.
- **MCP (Model Context Protocol)**.

## Building and Running

### Prerequisites:
- Docker and Docker Compose.
- `uv` Python package manager.

### Setup & Commands:
```bash
# Initial setup (generates secrets and initializes .env)
./start.sh

# Install dependencies
uv sync --group dev

# Run the OpenAI Proxy locally
# Backends: 'anthropic' (default, requires ANTHROPIC_API_KEY) or 'cli'
ANTHROPIC_API_KEY=sk-... uv run llm-mcp-proxy
# or
OPENAI_PROXY_BACKEND=cli uv run llm-mcp-proxy

# Manage Docker services
docker compose up -d        # Start the stack
docker compose stop         # Stop the stack
docker compose logs -f      # Follow logs
```

### Testing:
```bash
# Run all tests
uv run pytest

# Run proxy unit tests (no Docker required)
uv run pytest tests/test_openai_proxy.py

# Run MCP integration tests (requires Docker services running)
uv run pytest tests/mcp
```

## Development Conventions

- **Package Management**: Always use `uv`. Avoid `pip` directly.
- **Coding Style**:
    - Follow **PEP 8** and Python 3.12+ conventions.
    - 4-space indentation.
    - `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants.
    - Use type hints for public helpers and data structures.
- **Project Structure**:
    - Core logic: `src/llm_mcp_stack/`
    - Tests: `tests/`
    - Configuration: `.env` (managed via `start.sh` and `.env.example`).
- **Commits**: Follow **Conventional Commits** (e.g., `feat:`, `fix:`, `refactor:`).
- **Testing Practice**:
    - Use `pytest` and `pytest-asyncio`.
    - Unit tests should use mocks to avoid external API calls.
    - Integration tests that depend on the Docker stack go into `tests/mcp/`.
- **Security**: Never commit `.env` or secrets. Ensure `OPENAI_PROXY_API_KEY` is used if exposing the proxy.
