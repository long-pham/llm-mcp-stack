# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `src/llm_mcp_stack/` (mainly `openai_proxy.py`, exposed as `llm-mcp-proxy`).
Tests live in `tests/`:
- `tests/test_openai_proxy.py` for unit tests (no external services).
- `tests/mcp/` for MCP integration tests (requires Docker services running).
Container orchestration is in `docker-compose.yml`, with helper bootstrap in `start.sh`.
Environment defaults and required variables are documented in `.env.example`.

## Build, Test, and Development Commands
Use `uv` for dependency and run workflows (do not use `pip` directly).
- `uv sync --group dev`: install runtime + dev dependencies.
- `uv run llm-mcp-proxy`: run the OpenAI-compatible proxy locally (set `ANTHROPIC_API_KEY` first).
- `./start.sh`: initialize `.env`, generate missing secrets, and start Docker services.
- `docker compose up -d`: start stack when `.env` is already configured.
- `uv run pytest`: run all tests.
- `uv run pytest tests/test_openai_proxy.py`: run fast unit tests only.
- `uv run pytest tests/mcp`: run MCP integration tests (Docker required).

## Coding Style & Naming Conventions
Follow Python 3.12+ conventions with 4-space indentation and PEP 8 naming:
- `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, `Test...` classes in tests.
- Keep type hints for public helpers and request/response shaping logic.
- Group related helpers with brief section comments, consistent with `openai_proxy.py`.

## Testing Guidelines
Use `pytest` (with `pytest-asyncio` for async coverage). Name files `test_*.py` and test functions `test_*`.
Prefer isolated unit tests with mocks for Anthropic/CLI behavior; place service-dependent cases under `tests/mcp/`.
Before opening a PR, run at least `uv run pytest`; for stack changes, also run MCP tests.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit style: `feat:`, `fix:`, `refactor:` (example: `fix: remove hardcoded credentials and restrict CORS`).
Keep commits focused and scoped to one change set. PRs should include:
- Clear summary of behavior changes.
- Linked issue/context when relevant.
- Test evidence (commands run and results).
- Updated docs/config examples when changing env vars, ports, or service behavior.

## Security & Configuration Tips
Never commit `.env` or real secrets. Start from `.env.example` and rotate generated credentials for shared environments.
When exposing the proxy, set `OPENAI_PROXY_API_KEY` and restrict `OPENAI_PROXY_CORS_ORIGINS`.
