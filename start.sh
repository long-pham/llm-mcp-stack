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

# Load current values
source .env

# Auto-generate POSTGRES_PASSWORD if missing or placeholder
if [[ -z "$POSTGRES_PASSWORD" ]] || [[ "$POSTGRES_PASSWORD" == *"your-secure-password"* ]] || [[ "$POSTGRES_PASSWORD" == *"CHANGE_ME"* ]]; then
    echo "Generating POSTGRES_PASSWORD..."
    update_env "POSTGRES_PASSWORD" "$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"
fi

# Auto-generate BETTER_AUTH_SECRET if missing or placeholder
if [[ -z "$BETTER_AUTH_SECRET" ]] || [[ "$BETTER_AUTH_SECRET" == *"your-auth-secret"* ]] || [[ "$BETTER_AUTH_SECRET" == *"CHANGE_ME"* ]]; then
    echo "Generating BETTER_AUTH_SECRET..."
    update_env "BETTER_AUTH_SECRET" "$(openssl rand -base64 32)"
fi

# Auto-generate SEARXNG_SECRET_KEY if missing or placeholder
if [[ -z "$SEARXNG_SECRET_KEY" ]] || [[ "$SEARXNG_SECRET_KEY" == *"your-searxng-secret"* ]]; then
    echo "Generating SEARXNG_SECRET_KEY..."
    update_env "SEARXNG_SECRET_KEY" "$(openssl rand -hex 32)"
fi

# Reload after generating secrets
source .env

# Set default ports if not configured
SEARXNG_PORT=${SEARXNG_PORT:-38080}
SEARXNG_MCP_PORT=${SEARXNG_MCP_PORT:-38081}
CRAWL4AI_PORT=${CRAWL4AI_PORT:-11235}
METAMCP_PORT=${METAMCP_PORT:-12008}

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
echo "  - MetaMCP:        http://localhost:${METAMCP_PORT}"
echo ""
echo "To add to Claude Code:"
echo "  claude mcp add --transport sse crawl4ai http://localhost:${CRAWL4AI_PORT}/mcp/sse"
echo "  claude mcp add --transport http searxng http://localhost:${SEARXNG_MCP_PORT}/mcp"
echo ""
echo "Check status: docker compose ps"
echo "View logs:    docker compose logs -f"
