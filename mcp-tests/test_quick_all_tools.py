"""
Quick test script for all MetaMCP tools.
Tests each tool individually with proper timeout handling.

Usage:
    uv run python test_quick_all_tools.py
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

TOOL_TIMEOUT = 60.0  # Per-tool timeout
HTTP_TIMEOUT = httpx.Timeout(120.0, read=90.0)
METAMCP_API_KEY = os.getenv("METAMCP_API_KEY", "")
METAMCP_URL = os.getenv("METAMCP_URL", "http://opidev.local:12008/metamcp/general/mcp")

# All tools to test with minimal arguments
TOOL_TESTS = [
    ("crawl4ai__md", {"url": "https://example.com"}),
    ("crawl4ai__html", {"url": "https://example.com"}),
    ("crawl4ai__crawl", {"urls": ["https://example.com"]}),
    ("crawl4ai__ask", {}),
    ("crawl4ai__execute_js", {"url": "https://example.com", "scripts": ["document.title"]}),
    ("crawl4ai__screenshot", {"url": "https://example.com"}),
    ("crawl4ai__pdf", {"url": "https://example.com"}),
    ("SearXNG__searxng_web_search", {"query": "test"}),
    ("SearXNG__web_url_read", {"url": "https://example.com", "maxLength": 500}),
    ("DuckDuckGo__web-search", {"query": "test", "numResults": 2}),
    ("DuckDuckGo__iask-search", {"query": "hello"}),
    ("DuckDuckGo__brave-search", {"query": "test"}),
    ("DuckDuckGo__monica-search", {"query": "hi"}),
]


async def test_single_tool(tool_name: str, args: dict) -> tuple[str, str, float]:
    """Test a single tool in its own session."""
    headers = {"Authorization": f"Bearer {METAMCP_API_KEY}"} if METAMCP_API_KEY else {}
    start = time.time()

    try:
        async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as http_client:
            async with streamable_http_client(METAMCP_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments=args),
                        timeout=TOOL_TIMEOUT
                    )
                    elapsed = time.time() - start
                    content_count = len(result.content) if result.content else 0
                    return (tool_name, f"PASS ({content_count} items)", elapsed)
    except asyncio.TimeoutError:
        return (tool_name, "TIMEOUT", time.time() - start)
    except Exception as e:
        return (tool_name, f"FAIL: {type(e).__name__}", time.time() - start)


async def list_tools() -> list[str]:
    """List available tools."""
    headers = {"Authorization": f"Bearer {METAMCP_API_KEY}"} if METAMCP_API_KEY else {}

    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as http_client:
        async with streamable_http_client(METAMCP_URL, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [t.name for t in tools.tools]


async def main():
    print("Quick MetaMCP All Tools Test")
    print("=" * 60)

    # Get available tools
    try:
        available = set(await list_tools())
        print(f"Found {len(available)} tools\n")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

    results = []
    start_total = time.time()

    # Test each tool
    for tool_name, args in TOOL_TESTS:
        if tool_name not in available:
            print(f"⚠ {tool_name}: NOT AVAILABLE")
            results.append((tool_name, "SKIP", 0))
            continue

        name, status, elapsed = await test_single_tool(tool_name, args)
        icon = "✓" if "PASS" in status else "✗" if "FAIL" in status or "TIMEOUT" in status else "⚠"
        print(f"{icon} {name}: {status} ({elapsed:.1f}s)")
        results.append((name, status, elapsed))

    total_time = time.time() - start_total

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    failed = sum(1 for _, s, _ in results if "FAIL" in s or "TIMEOUT" in s)
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
