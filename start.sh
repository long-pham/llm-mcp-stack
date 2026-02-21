#!/bin/bash
# Start all LLM MCP services
#
# Usage: ./start.sh [--force-recreate] [--build] [--pull]
#   --force-recreate  Recreate containers even if config hasn't changed
#   --build           Rebuild images before starting
#   --pull            Pull latest images before starting

set -e

cd "$(dirname "$0")"

# Parse flags
COMPOSE_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --force-recreate) COMPOSE_FLAGS+=(--force-recreate) ;;
        --build)          COMPOSE_FLAGS+=(--build) ;;
        --pull)           COMPOSE_FLAGS+=(--pull always) ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--force-recreate] [--build] [--pull]"
            exit 1
            ;;
    esac
done

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Helper function to update or append env var
update_env() {
    local key=$1
    local value=$2
    # Use | as sed delimiter to avoid issues with / in base64 values
    if grep -q "^${key}=" .env; then
        sed -i.bak "s|^${key}=.*|${key}=${value}|" .env && rm -f .env.bak
    else
        echo "${key}=${value}" >> .env
    fi
}

# Load .env values safely (without executing shell code).
# Note: does not handle escaped quotes inside values (e.g. "val\"ue").
load_env_file() {
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments/blank lines.
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue

        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"

            # Strip matching surrounding quotes if present.
            if [[ "$value" =~ ^\"(.*)\"$ ]]; then
                value="${BASH_REMATCH[1]}"
            elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi

            export "${key}=${value}"
        fi
    done < .env
}

# Load current values
load_env_file

# Auto-generate SEARXNG_SECRET_KEY if missing or placeholder
if [[ -z "$SEARXNG_SECRET_KEY" ]] || [[ "$SEARXNG_SECRET_KEY" == *"your-searxng-secret"* ]]; then
    echo "Generating SEARXNG_SECRET_KEY..."
    update_env "SEARXNG_SECRET_KEY" "$(openssl rand -hex 32)"
fi

# Reload after generating secrets
load_env_file

# Set default ports if not configured
SEARXNG_MCP_PORT=${SEARXNG_MCP_PORT:-38081}
CRAWL4AI_PORT=${CRAWL4AI_PORT:-11235}
CRAWL4AI_MAX_TASKS=${CRAWL4AI_MAX_TASKS:-10}
PAPER_SEARCH_PORT=${PAPER_SEARCH_PORT:-38082}
MCPHUB_PORT=${MCPHUB_PORT:-3000}
OPENAI_PROXY_PORT=${OPENAI_PROXY_PORT:-8090}

# Start all services
echo "Starting LLM MCP services..."
docker compose up -d "${COMPOSE_FLAGS[@]}"

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
wait_healthy() {
    local service=$1
    local max_wait=60
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "missing")
        if [ "$status" = "healthy" ]; then
            echo "  $service: healthy"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "  $service: timed out waiting for healthy (status: $status)"
    return 1
}

wait_healthy searxng
wait_healthy searxng-mcp
wait_healthy crawl4ai
wait_healthy paper-search
wait_healthy mcphub

echo ""
echo "Services started! Endpoints:"
echo "  - SearXNG MCP:    http://localhost:${SEARXNG_MCP_PORT}/mcp"
echo "  - Crawl4AI MCP:   http://localhost:${CRAWL4AI_PORT}/mcp/sse"
echo "  - Paper Search:   http://localhost:${PAPER_SEARCH_PORT}/sse"
echo "  - MCPHub:         http://localhost:${MCPHUB_PORT}"
echo ""
echo "To add to Claude Code (aggregated via MCPHub):"
echo "  claude mcp add --transport sse --scope user  mcphub http://localhost:${MCPHUB_PORT}/mcp"
echo ""
echo "Or add individual servers:"
echo "  claude mcp add --transport sse --scope user crawl4ai http://localhost:${CRAWL4AI_PORT}/mcp/sse"
echo "  claude mcp add --transport http --scope user searxng http://localhost:${SEARXNG_MCP_PORT}/mcp"
echo "  claude mcp add --transport sse --scope user paper-search http://localhost:${PAPER_SEARCH_PORT}/sse"
echo ""
echo "Check status: docker compose ps"
echo "View logs:    docker compose logs -f"
