#!/bin/bash
# Start all LLM MCP services

set -e

cd "$(dirname "$0")"

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
SEARXNG_PORT=${SEARXNG_PORT:-38080}
SEARXNG_MCP_PORT=${SEARXNG_MCP_PORT:-38081}
CRAWL4AI_PORT=${CRAWL4AI_PORT:-11235}
MCPHUB_PORT=${MCPHUB_PORT:-3000}

# Start all services
echo "Starting LLM MCP services..."
docker compose up -d

# Wait for health checks
echo ""
echo "Waiting for services to be healthy..."
sleep 5

echo ""
echo "Services started! Endpoints:"
echo "  - SearXNG:        http://localhost:${SEARXNG_PORT}"
echo "  - SearXNG MCP:    http://localhost:${SEARXNG_MCP_PORT}/mcp"
echo "  - Crawl4AI:       http://localhost:${CRAWL4AI_PORT}"
echo "  - Crawl4AI MCP:   http://localhost:${CRAWL4AI_PORT}/mcp/sse"
echo "  - MCPHub:         http://localhost:${MCPHUB_PORT}"
echo ""
echo "To add to Claude Code (aggregated via MCPHub):"
echo "  claude mcp add --transport sse mcphub http://localhost:${MCPHUB_PORT}/mcp"
echo ""
echo "Or add individual servers:"
echo "  claude mcp add --transport sse crawl4ai http://localhost:${CRAWL4AI_PORT}/mcp/sse"
echo "  claude mcp add --transport http searxng http://localhost:${SEARXNG_MCP_PORT}/mcp"
echo ""
echo "Check status: docker compose ps"
echo "View logs:    docker compose logs -f"
