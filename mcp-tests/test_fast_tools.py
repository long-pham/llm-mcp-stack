"""
Fast test for MetaMCP tools - only tests quick/reliable tools.

Usage:
    uv run python test_fast_tools.py
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

TOOL_TIMEOUT = 30.0  # Shorter timeout for fast tools
HTTP_TIMEOUT = httpx.Timeout(60.0, read=45.0)
METAMCP_API_KEY = os.getenv("METAMCP_API_KEY", "")
METAMCP_URL = os.getenv("METAMCP_URL", "http://opidev.local:12008/metamcp/general/mcp")

# Only fast/reliable tools
FAST_TOOLS = [
    ("crawl4ai__md", {"url": "https://example.com"}),
    ("crawl4ai__crawl", {"urls": ["https://example.com"]}),
    ("DuckDuckGo__brave-search", {"query": "test"}),
]


async def main():
    print("Fast MetaMCP Tools Test")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {METAMCP_API_KEY}"} if METAMCP_API_KEY else {}
    results = []
    start_total = time.time()

    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as http_client:
        async with streamable_http_client(METAMCP_URL, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                available = {t.name for t in tools.tools}
                print(f"Available: {len(available)} tools")
                print(f"Testing: {len(FAST_TOOLS)} fast tools\n")

                for tool_name, args in FAST_TOOLS:
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
                        preview = ""
                        if result.content and hasattr(result.content[0], 'text'):
                            text = result.content[0].text
                            preview = f" - {len(text)} chars"

                        print(f"✓ {tool_name}: OK ({content_count} items{preview}) [{elapsed:.1f}s]")
                        results.append((tool_name, "PASS", elapsed))
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start
                        print(f"✗ {tool_name}: TIMEOUT [{elapsed:.1f}s]")
                        results.append((tool_name, "TIMEOUT", elapsed))
                    except Exception as e:
                        elapsed = time.time() - start
                        print(f"✗ {tool_name}: {type(e).__name__} [{elapsed:.1f}s]")
                        results.append((tool_name, f"FAIL", elapsed))

    total_time = time.time() - start_total

    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s not in ["PASS", "SKIP"])
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
