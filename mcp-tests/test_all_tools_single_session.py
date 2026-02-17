"""
Test ALL MetaMCP tools in a single session.

Usage:
    uv run python test_all_tools_single_session.py
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

TOOL_TIMEOUT = 90.0  # Per-tool timeout (some AI searches are slow)
HTTP_TIMEOUT = httpx.Timeout(180.0, read=120.0)
METAMCP_API_KEY = os.getenv("METAMCP_API_KEY", "")
METAMCP_URL = os.getenv("METAMCP_URL", "http://opidev.local:12008/metamcp/general/mcp")

# All tools with minimal test arguments
ALL_TOOLS = [
    # Crawl4AI tools
    ("crawl4ai__md", {"url": "https://example.com"}),
    ("crawl4ai__html", {"url": "https://example.com"}),
    ("crawl4ai__crawl", {"urls": ["https://example.com"]}),
    ("crawl4ai__ask", {}),
    ("crawl4ai__execute_js", {"url": "https://example.com", "scripts": ["document.title"]}),
    ("crawl4ai__screenshot", {"url": "https://example.com"}),
    ("crawl4ai__pdf", {"url": "https://example.com"}),
    # SearXNG tools
    ("SearXNG__searxng_web_search", {"query": "test"}),
    ("SearXNG__web_url_read", {"url": "https://example.com", "maxLength": 500}),
]


async def main():
    print("MetaMCP All Tools Test (Single Session)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {METAMCP_API_KEY}"} if METAMCP_API_KEY else {}
    results = []
    start_total = time.time()

    try:
        async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as http_client:
            async with streamable_http_client(METAMCP_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    available = {t.name for t in tools.tools}
                    print(f"Available: {len(available)} tools")
                    print(f"Testing: {len(ALL_TOOLS)} tools\n")

                    for tool_name, args in ALL_TOOLS:
                        if tool_name not in available:
                            print(f"⚠ {tool_name}: NOT AVAILABLE")
                            results.append((tool_name, "SKIP", 0))
                            continue

                        start = time.time()
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, arguments=args),
                                timeout=TOOL_TIMEOUT
                            )
                            elapsed = time.time() - start
                            content_count = len(result.content) if result.content else 0

                            # Show preview of result
                            chars = 0
                            if result.content and hasattr(result.content[0], 'text'):
                                chars = len(result.content[0].text)

                            print(f"✓ {tool_name}: {content_count} items, {chars} chars [{elapsed:.1f}s]")
                            results.append((tool_name, "PASS", elapsed))
                        except asyncio.TimeoutError:
                            elapsed = time.time() - start
                            print(f"✗ {tool_name}: TIMEOUT [{elapsed:.1f}s]")
                            results.append((tool_name, "TIMEOUT", elapsed))
                        except Exception as e:
                            elapsed = time.time() - start
                            err = str(e)[:50] if len(str(e)) > 50 else str(e)
                            print(f"✗ {tool_name}: {type(e).__name__} - {err} [{elapsed:.1f}s]")
                            results.append((tool_name, "FAIL", elapsed))

    except Exception as e:
        print(f"Session error: {type(e).__name__}: {e}")
        return False

    total_time = time.time() - start_total

    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s not in ["PASS", "SKIP"])
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_time:.1f}s")

    if results:
        avg_time = sum(t for _, _, t in results if t > 0) / len([t for _, _, t in results if t > 0])
        print(f"Average time per tool: {avg_time:.1f}s")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
