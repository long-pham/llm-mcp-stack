"""
Pytest tests for ALL MCPHub tools.

Tests all tools available through MCPHub aggregator.

Usage:
    uv run pytest test_mcphub_all_tools.py -v
    uv run pytest test_mcphub_all_tools.py -v -k crawl4ai
    uv run pytest test_mcphub_all_tools.py -v -k searxng
"""

import asyncio
import os
from contextlib import asynccontextmanager

import httpx
import pytest
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

# Configuration
TOOL_TIMEOUT = 90.0
HTTP_TIMEOUT = httpx.Timeout(180.0, read=120.0)
MCPHUB_URL = os.getenv("MCPHUB_URL", "http://localhost:3000/sse")
HEALTH_URL = os.getenv("MCPHUB_HEALTH_URL", "http://localhost:3000")


@asynccontextmanager
async def mcphub_session():
    """Create an MCP client session for MCPHub."""
    async with sse_client(MCPHUB_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def check_health() -> bool:
    """Check if MCPHub is healthy."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(HEALTH_URL)
            return response.status_code == 200
        except httpx.RequestError:
            return False


class TestCrawl4AITools:
    """Test all Crawl4AI tools."""

    async def test_crawl4ai_md(self):
        """crawl4ai__md - Fetch URL as markdown."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__md", arguments={"url": "https://example.com"}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__md: {len(result.content)} items")

    async def test_crawl4ai_html(self):
        """crawl4ai__html - Fetch raw HTML."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__html", arguments={"url": "https://example.com"}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__html: {len(result.content)} items")

    async def test_crawl4ai_crawl(self):
        """crawl4ai__crawl - Crawl URLs."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__crawl", arguments={"urls": ["https://example.com"]}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__crawl: {len(result.content)} items")

    async def test_crawl4ai_ask(self):
        """crawl4ai__ask - Ask about Crawl4AI library."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__ask", arguments={}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__ask: {len(result.content)} items")

    async def test_crawl4ai_execute_js(self):
        """crawl4ai__execute_js - Execute JavaScript on page."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool(
                    "crawl4ai__execute_js",
                    arguments={"url": "https://example.com", "scripts": ["document.title"]}
                ),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__execute_js: {len(result.content)} items")

    async def test_crawl4ai_screenshot(self):
        """crawl4ai__screenshot - Take page screenshot."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__screenshot", arguments={"url": "https://example.com"}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__screenshot: {len(result.content)} items")

    async def test_crawl4ai_pdf(self):
        """crawl4ai__pdf - Generate PDF from page."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("crawl4ai__pdf", arguments={"url": "https://example.com"}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\ncrawl4ai__pdf: {len(result.content)} items")


class TestSearXNGTools:
    """Test SearXNG tools (may timeout if server issues)."""

    @pytest.mark.timeout(120)
    async def test_searxng_web_search(self):
        """searxng__searxng_web_search - Web search."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool("searxng__searxng_web_search", arguments={"query": "test"}),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\nsearxng__searxng_web_search: {len(result.content)} items")

    @pytest.mark.timeout(120)
    async def test_searxng_web_url_read(self):
        """searxng__web_url_read - Read URL content."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            result = await asyncio.wait_for(
                session.call_tool(
                    "searxng__web_url_read",
                    arguments={"url": "https://example.com", "maxLength": 500}
                ),
                timeout=TOOL_TIMEOUT
            )
            assert result.content is not None
            print(f"\nsearxng__web_url_read: {len(result.content)} items")


class TestToolDiscovery:
    """Test tool discovery and listing."""

    async def test_list_all_tools(self):
        """List all available tools in MCPHub."""
        if not await check_health():
            pytest.skip("MCPHub not healthy")

        async with mcphub_session() as session:
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]

            print(f"\nMCPHub Tools ({len(tool_names)} total):")
            for name in tool_names:
                print(f"  - {name}")

            assert len(tool_names) >= 10, "Expected at least 10 tools"


if __name__ == "__main__":
    import sys

    async def quick_test():
        """Quick test of all tools."""
        print("Quick MCPHub Tools Test")
        print("=" * 60)

        if not await check_health():
            print("MCPHub is not healthy!")
            return 1

        async with mcphub_session() as session:
            tools = await session.list_tools()
            print(f"Found {len(tools.tools)} tools:")
            for t in tools.tools:
                print(f"  - {t.name}")

        return 0

    sys.exit(asyncio.run(quick_test()))
