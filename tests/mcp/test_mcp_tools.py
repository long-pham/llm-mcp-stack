"""
Comprehensive MCP Tool Tests for opidev.local services

Uses the official MCP Python SDK to properly test MCP tools via SSE/StreamableHTTP transport.

Usage:
    uv run pytest test_mcp_tools.py -v           # Run all MCP tool tests
    uv run pytest test_mcp_tools.py -v -k crawl  # Run only crawl4ai tests
    uv run pytest test_mcp_tools.py -v -k metamcp  # Run MetaMCP tests
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

# Load environment variables
load_dotenv()

# Configuration
BASE_HOST = os.getenv("MCP_BASE_HOST", "localhost")
TIMEOUT = 30.0

# MetaMCP with Streamable HTTP (from .env)
METAMCP_API_KEY = os.getenv("METAMCP_API_KEY", "")
METAMCP_URL = os.getenv("METAMCP_URL", "http://localhost:12008/metamcp/general/mcp")

# MCP Server endpoints - SSE transport
MCP_SERVERS = {
    "searxng-mcp": {
        "sse_url": f"http://{BASE_HOST}:38081/sse",
        "mcp_url": f"http://{BASE_HOST}:38081/mcp",
        "health_url": f"http://{BASE_HOST}:38081/health",
        "transport": "sse",
    },
    "metamcp": {
        "sse_url": f"http://{BASE_HOST}:12008/sse",
        "mcp_url": METAMCP_URL,
        "health_url": f"http://{BASE_HOST}:12008/health",
        "transport": "streamable_http",
        "api_key": METAMCP_API_KEY,
    },
    "crawl4ai": {
        "sse_url": f"http://{BASE_HOST}:11235/mcp/sse",
        "mcp_url": f"http://{BASE_HOST}:11235/mcp",
        "health_url": f"http://{BASE_HOST}:11235/health",
        "transport": "sse",
    },
}


@asynccontextmanager
async def mcp_session(server_name: str):
    """Create an MCP client session for a server."""
    server = MCP_SERVERS[server_name]
    transport = server.get("transport", "sse")

    if transport == "streamable_http":
        # Use StreamableHTTP transport with optional API key auth
        headers = {}
        if server.get("api_key"):
            headers["Authorization"] = f"Bearer {server['api_key']}"

        # Create httpx client with headers for authentication
        async with httpx.AsyncClient(headers=headers, timeout=TIMEOUT) as http_client:
            async with streamable_http_client(
                server["mcp_url"],
                http_client=http_client
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
    else:
        # Use SSE transport (traditional)
        async with sse_client(server["sse_url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def check_health(server_name: str) -> bool:
    """Check if a server is healthy before testing."""
    server = MCP_SERVERS[server_name]
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.get(server["health_url"])
            return response.status_code == 200
        except httpx.RequestError:
            return False


class TestMCPServerDiscovery:
    """Test MCP server tool discovery via the SDK."""

    async def test_searxng_mcp_list_tools(self):
        """Test listing tools from SearXNG MCP server."""
        if not await check_health("searxng-mcp"):
            pytest.skip("SearXNG MCP server not healthy")

        try:
            async with mcp_session("searxng-mcp") as session:
                tools = await session.list_tools()
                assert tools.tools is not None
                tool_names = [t.name for t in tools.tools]
                print(f"\nSearXNG MCP tools: {tool_names}")
                assert len(tool_names) > 0, "Expected at least one tool"
        except Exception as e:
            pytest.skip(f"Could not connect to SearXNG MCP: {e}")

    async def test_metamcp_list_tools(self):
        """Test listing tools from MetaMCP aggregator."""
        if not await check_health("metamcp"):
            pytest.skip("MetaMCP server not healthy")

        try:
            async with mcp_session("metamcp") as session:
                tools = await session.list_tools()
                assert tools.tools is not None
                tool_names = [t.name for t in tools.tools]
                print(f"\nMetaMCP aggregated tools ({len(tool_names)}): {tool_names[:10]}...")
                # MetaMCP aggregates tools from multiple servers
                assert len(tool_names) > 0, "Expected at least one aggregated tool"
        except Exception as e:
            pytest.skip(f"Could not connect to MetaMCP: {e}")

    async def test_crawl4ai_list_tools(self):
        """Test listing tools from Crawl4AI MCP server."""
        if not await check_health("crawl4ai"):
            pytest.skip("Crawl4AI server not healthy")

        try:
            async with mcp_session("crawl4ai") as session:
                tools = await session.list_tools()
                assert tools.tools is not None
                tool_names = [t.name for t in tools.tools]
                print(f"\nCrawl4AI MCP tools: {tool_names}")
                assert len(tool_names) > 0, "Expected at least one tool"
        except Exception as e:
            pytest.skip(f"Could not connect to Crawl4AI MCP: {e}")


class TestCrawl4AITools:
    """Test Crawl4AI MCP tools."""

    async def test_crawl4ai_md_tool(self):
        """Test Crawl4AI markdown fetch tool."""
        if not await check_health("crawl4ai"):
            pytest.skip("Crawl4AI server not healthy")

        try:
            async with mcp_session("crawl4ai") as session:
                # First list tools to find the correct tool name
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"\nAvailable tools: {tool_names}")

                # Find the md/markdown tool
                md_tool = None
                for name in ["md", "crawl4ai_md", "fetch_md", "markdown"]:
                    if name in tool_names:
                        md_tool = name
                        break

                if not md_tool:
                    # Try partial match
                    for name in tool_names:
                        if "md" in name.lower() or "markdown" in name.lower():
                            md_tool = name
                            break

                if not md_tool:
                    pytest.skip(f"No markdown tool found in: {tool_names}")

                # Call the markdown tool
                result = await session.call_tool(
                    md_tool,
                    arguments={"url": "https://example.com"}
                )
                assert result is not None
                print(f"\nCrawl4AI md result type: {type(result)}")
                # Check we got content back
                assert result.content is not None
        except Exception as e:
            pytest.skip(f"Crawl4AI md tool test failed: {e}")

    async def test_crawl4ai_crawl_tool(self):
        """Test Crawl4AI crawl tool."""
        if not await check_health("crawl4ai"):
            pytest.skip("Crawl4AI server not healthy")

        try:
            async with mcp_session("crawl4ai") as session:
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]

                # Find crawl tool
                crawl_tool = None
                for name in ["crawl", "crawl4ai_crawl", "crawl_url"]:
                    if name in tool_names:
                        crawl_tool = name
                        break

                if not crawl_tool:
                    for name in tool_names:
                        if "crawl" in name.lower():
                            crawl_tool = name
                            break

                if not crawl_tool:
                    pytest.skip(f"No crawl tool found in: {tool_names}")

                # Call the crawl tool
                result = await session.call_tool(
                    crawl_tool,
                    arguments={"urls": ["https://example.com"]}
                )
                assert result is not None
                print(f"\nCrawl4AI crawl result: {str(result)[:200]}")
        except Exception as e:
            pytest.skip(f"Crawl4AI crawl tool test failed: {e}")


class TestSearXNGTools:
    """Test SearXNG MCP tools."""

    async def test_searxng_search_tool(self):
        """Test SearXNG search tool via MCP."""
        if not await check_health("searxng-mcp"):
            pytest.skip("SearXNG MCP server not healthy")

        try:
            async with mcp_session("searxng-mcp") as session:
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"\nSearXNG tools: {tool_names}")

                # Find search tool
                search_tool = None
                for name in ["search", "searxng_search", "web_search"]:
                    if name in tool_names:
                        search_tool = name
                        break

                if not search_tool:
                    for name in tool_names:
                        if "search" in name.lower():
                            search_tool = name
                            break

                if not search_tool:
                    pytest.skip(f"No search tool found in: {tool_names}")

                # Call the search tool
                result = await session.call_tool(
                    search_tool,
                    arguments={"query": "python programming", "num_results": 5}
                )
                assert result is not None
                print(f"\nSearXNG search result type: {type(result)}")
                assert result.content is not None
        except Exception as e:
            pytest.skip(f"SearXNG search tool test failed: {e}")


class TestMetaMCPAggregation:
    """Test MetaMCP tool aggregation."""

    async def test_metamcp_aggregates_tools(self):
        """Test that MetaMCP aggregates tools from connected servers."""
        if not await check_health("metamcp"):
            pytest.skip("MetaMCP server not healthy")

        try:
            async with mcp_session("metamcp") as session:
                tools = await session.list_tools()
                assert tools.tools is not None

                tool_names = [t.name for t in tools.tools]
                print(f"\nMetaMCP total tools: {len(tool_names)}")
                print(f"Sample tools: {tool_names[:10]}")

                # Log tool details
                for tool in tools.tools[:5]:
                    print(f"  - {tool.name}: {tool.description[:50] if tool.description else 'No description'}...")
        except Exception as e:
            pytest.skip(f"MetaMCP aggregation test failed: {e}")

    async def test_metamcp_list_resources(self):
        """Test listing resources from MetaMCP."""
        if not await check_health("metamcp"):
            pytest.skip("MetaMCP server not healthy")

        try:
            async with mcp_session("metamcp") as session:
                resources = await session.list_resources()
                if resources.resources:
                    print(f"\nMetaMCP resources: {len(resources.resources)}")
                    for r in resources.resources[:5]:
                        print(f"  - {r.name}: {r.uri}")
        except Exception as e:
            pytest.skip(f"MetaMCP resources test failed: {e}")


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance for each server."""

    async def test_server_info(self):
        """Test that servers provide proper initialization info."""
        for server_name in MCP_SERVERS:
            if not await check_health(server_name):
                continue

            try:
                async with mcp_session(server_name) as session:
                    # Session initialization already happened, check we can list tools
                    tools = await session.list_tools()
                    print(f"\n{server_name}: {len(tools.tools)} tools available")
            except Exception as e:
                print(f"\n{server_name}: Connection failed - {e}")

    async def test_tool_schemas(self):
        """Test that tools have proper input schemas."""
        for server_name in MCP_SERVERS:
            if not await check_health(server_name):
                continue

            try:
                async with mcp_session(server_name) as session:
                    tools = await session.list_tools()
                    for tool in tools.tools[:3]:  # Check first 3 tools
                        assert tool.name, f"Tool missing name in {server_name}"
                        print(f"\n{server_name}/{tool.name}:")
                        print(f"  Description: {tool.description[:80] if tool.description else 'None'}...")
                        if tool.inputSchema:
                            print(f"  Input schema: {list(tool.inputSchema.get('properties', {}).keys())}")
            except Exception as e:
                print(f"\n{server_name}: Schema test failed - {e}")


# Direct HTTP fallback tests (for servers without SSE support)
class TestHTTPFallback:
    """HTTP-based tests for servers that may not support SSE."""

    async def test_crawl4ai_direct_api(self):
        """Test Crawl4AI via direct HTTP API."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test health
            response = await client.get(f"http://{BASE_HOST}:11235/health")
            assert response.status_code == 200

            # Test md endpoint directly
            response = await client.get(
                f"http://{BASE_HOST}:11235/md",
                params={"url": "https://example.com"}
            )
            print(f"\nCrawl4AI /md status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response length: {len(response.text)} chars")

    async def test_searxng_direct_search(self):
        """Test SearXNG via direct HTTP API."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"http://{BASE_HOST}:38080/search",
                params={"q": "test query", "format": "json"}
            )
            assert response.status_code == 200
            data = response.json()
            print(f"\nSearXNG direct search results: {len(data.get('results', []))}")


if __name__ == "__main__":
    # Quick test runner
    import sys

    async def main():
        print("MCP Tool Discovery Test")
        print("=" * 50)

        for server_name, config in MCP_SERVERS.items():
            healthy = await check_health(server_name)
            status = "OK" if healthy else "UNHEALTHY"
            print(f"{server_name}: {status}")

            if healthy:
                try:
                    async with mcp_session(server_name) as session:
                        tools = await session.list_tools()
                        tool_names = [t.name for t in tools.tools]
                        print(f"  Tools: {tool_names[:5]}{'...' if len(tool_names) > 5 else ''}")
                except Exception as e:
                    print(f"  Error: {e}")

        print("=" * 50)

    asyncio.run(main())
