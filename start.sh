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
    if grep -q "^${key}=" .env; then
        sed -i.bak "s/^${key}=.*/${key}=${value}/" .env && rm -f .env.bak
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

# Start all services
echo "Starting LLM MCP services..."
docker compose up -d

# Wait for health checks
echo ""
echo "Waiting for services to be healthy..."
sleep 5

echo ""
echo "Services started! Endpoints:"
echo "  - SearXNG:        http://localhost:38080"
echo "  - SearXNG MCP:    http://localhost:38081/mcp"
echo "  - Crawl4AI:       http://localhost:11235"
echo "  - Crawl4AI MCP:   http://localhost:11235/mcp/sse"
echo "  - DuckDuckGo MCP: http://localhost:38020"
echo "  - MetaMCP:        http://localhost:12008"
echo ""
echo "To add to Claude Code:"
echo "  claude mcp add --transport sse crawl4ai http://localhost:11235/mcp/sse"
echo "  claude mcp add --transport http searxng http://localhost:38081/mcp"
echo "  claude mcp add --transport http duckduckgo http://localhost:38020/mcp"
echo ""
echo "Check status: docker compose ps"
echo "View logs:    docker compose logs -f"
