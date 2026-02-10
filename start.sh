#!/bin/bash
# Start all LLM MCP services

set -e

cd "$(dirname "$0")"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "WARNING: .env created from template. Please review and update:"
    echo "  - POSTGRES_PASSWORD (generate: openssl rand -base64 24)"
    echo "  - BETTER_AUTH_SECRET (generate: openssl rand -base64 32)"
    echo ""
    echo "SEARXNG_SECRET_KEY will be auto-generated."
    read -p "Press Enter to continue or Ctrl+C to edit .env first..."
fi

# Validate and auto-generate secrets
source .env

# Auto-generate SEARXNG_SECRET_KEY if missing or placeholder
if [[ -z "$SEARXNG_SECRET_KEY" ]] || [[ "$SEARXNG_SECRET_KEY" == *"your-searxng-secret"* ]]; then
    NEW_SECRET=$(openssl rand -hex 32)
    echo "Generating SEARXNG_SECRET_KEY..."
    if grep -q "^SEARXNG_SECRET_KEY=" .env; then
        sed -i.bak "s/^SEARXNG_SECRET_KEY=.*/SEARXNG_SECRET_KEY=$NEW_SECRET/" .env && rm -f .env.bak
    else
        echo "SEARXNG_SECRET_KEY=$NEW_SECRET" >> .env
    fi
    source .env
fi

# Validate other required secrets
if [[ "$POSTGRES_PASSWORD" == *"CHANGE_ME"* ]] || [[ "$POSTGRES_PASSWORD" == *"your-secure-password"* ]] || \
   [[ "$BETTER_AUTH_SECRET" == *"CHANGE_ME"* ]] || [[ "$BETTER_AUTH_SECRET" == *"your-auth-secret"* ]]; then
    echo "ERROR: Please update required secrets in .env"
    echo ""
    echo "Required secrets:"
    echo "  POSTGRES_PASSWORD:   openssl rand -base64 24"
    echo "  BETTER_AUTH_SECRET:  openssl rand -base64 32"
    exit 1
fi

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
