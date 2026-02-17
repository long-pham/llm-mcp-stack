"""
Test suite for MCP services running on opidev.local

Usage:
    uv run pytest                    # Run all tests
    uv run pytest -m "not slow"      # Skip slow tests
    uv run mcp-check                 # Quick health check
    uv run python test_mcp_services.py  # Quick health check (alt)
"""

import os

import httpx

try:
    import pytest
except ImportError:
    pytest = None  # Allow running standalone without pytest

# Configuration - adjust host if needed
BASE_HOST = os.getenv("MCP_BASE_HOST", "localhost")
TIMEOUT = 10.0


def _endpoint_reachable(url: str) -> bool:
    try:
        with httpx.Client(timeout=1.5) as client:
            response = client.get(url)
            return response.status_code < 500
    except httpx.RequestError:
        return False


if pytest and not all(
    [
        _endpoint_reachable(f"http://{BASE_HOST}:38080/healthz"),
        _endpoint_reachable(f"http://{BASE_HOST}:38081/health"),
        _endpoint_reachable(f"http://{BASE_HOST}:11235/health"),
        _endpoint_reachable(f"http://{BASE_HOST}:12008/health"),
    ]
):
    pytestmark = pytest.mark.skip(
        reason="MCP integration services are not reachable; start docker compose first"
    )


class TestSearXNG:
    """Tests for SearXNG metasearch engine."""

    base_url = f"http://{BASE_HOST}:38080"

    def test_health_endpoint(self):
        """Test SearXNG health endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/healthz")
            assert response.status_code == 200, f"SearXNG health check failed: {response.text}"

    def test_homepage_loads(self):
        """Test SearXNG homepage is accessible."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(self.base_url)
            assert response.status_code == 200
            assert "searx" in response.text.lower() or "search" in response.text.lower()

    def test_search_endpoint(self):
        """Test SearXNG can perform a basic search."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data or "query" in data


class TestSearXNGMCP:
    """Tests for SearXNG MCP Server."""

    base_url = f"http://{BASE_HOST}:38081"

    def test_health_endpoint(self):
        """Test SearXNG MCP health endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/health")
            assert response.status_code == 200, (
                f"SearXNG MCP health check failed: {response.status_code}"
            )

    def test_mcp_endpoint_exists(self):
        """Test MCP endpoint exists (400 = needs proper MCP request)."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/mcp")
            # 400 means endpoint exists but needs proper MCP protocol
            assert response.status_code in [200, 400, 405], (
                f"SearXNG MCP endpoint not found: {response.status_code}"
            )


class TestCrawl4AI:
    """Tests for Crawl4AI web crawling service."""

    base_url = f"http://{BASE_HOST}:11235"

    def test_health_endpoint(self):
        """Test Crawl4AI health endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/health")
            assert response.status_code == 200, f"Crawl4AI health check failed: {response.text}"

    def test_crawl_endpoint_exists(self):
        """Test Crawl4AI crawl endpoint is available."""
        with httpx.Client(timeout=TIMEOUT) as client:
            # 405 Method Not Allowed means endpoint exists but needs POST
            response = client.get(f"{self.base_url}/crawl")
            assert response.status_code in [200, 400, 405, 422], (
                f"Crawl4AI crawl endpoint check failed: {response.status_code}"
            )

    def test_md_endpoint(self):
        """Test Crawl4AI markdown endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/md")
            assert response.status_code in [200, 400, 405, 422], (
                f"Crawl4AI md endpoint check failed: {response.status_code}"
            )


class TestMetaMCP:
    """Tests for MetaMCP aggregator/gateway."""

    base_url = f"http://{BASE_HOST}:12008"

    def test_health_endpoint(self):
        """Test MetaMCP health endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/health")
            assert response.status_code == 200, f"MetaMCP health check failed: {response.status_code}"

    def test_homepage_redirect(self):
        """Test MetaMCP homepage redirects (Next.js app)."""
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            response = client.get(self.base_url)
            assert response.status_code == 200, f"MetaMCP homepage failed: {response.status_code}"

    def test_sse_endpoint_exists(self):
        """Test MetaMCP SSE endpoint exists."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/sse")
            # 307 redirect is expected
            assert response.status_code in [200, 307, 400, 405], (
                f"MetaMCP SSE endpoint check failed: {response.status_code}"
            )

    def test_mcp_endpoint_exists(self):
        """Test MetaMCP MCP endpoint exists."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(f"{self.base_url}/mcp")
            assert response.status_code in [200, 307, 400, 405], (
                f"MetaMCP MCP endpoint check failed: {response.status_code}"
            )


class TestConnectivity:
    """Cross-service connectivity tests."""

    def test_all_health_endpoints(self):
        """Test all service health endpoints pass."""
        health_checks = [
            ("SearXNG", f"http://{BASE_HOST}:38080/healthz"),
            ("SearXNG-MCP", f"http://{BASE_HOST}:38081/health"),
            ("Crawl4AI", f"http://{BASE_HOST}:11235/health"),
            ("MetaMCP", f"http://{BASE_HOST}:12008/health"),
        ]

        results = []
        with httpx.Client(timeout=TIMEOUT) as client:
            for name, url in health_checks:
                try:
                    response = client.get(url)
                    results.append((name, response.status_code == 200, response.status_code))
                except httpx.RequestError as e:
                    results.append((name, False, str(e)))

        # Print summary
        print("\n" + "=" * 50)
        print("Health Check Summary")
        print("=" * 50)
        for name, success, status in results:
            status_icon = "✓" if success else "✗"
            print(f"{status_icon} {name}: {status}")
        print("=" * 50)

        # Assert all health checks pass
        failed = [r for r in results if not r[1]]
        assert not failed, f"Health checks failed: {[f'{r[0]}={r[2]}' for r in failed]}"


class TestFunctionality:
    """Functional tests for services."""

    def test_searxng_search(self):
        """Test SearXNG can perform an actual search."""
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"http://{BASE_HOST}:38080/search",
                params={"q": "python programming", "format": "json"},
            )
            assert response.status_code == 200
            data = response.json()
            # Verify we got results
            assert "results" in data
            print(f"\nSearXNG returned {len(data.get('results', []))} results")

    @pytest.mark.slow if pytest else lambda f: f
    def test_crawl4ai_basic_crawl(self):
        """Test Crawl4AI can crawl a simple page (may be slow)."""
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"http://{BASE_HOST}:11235/crawl",
                json={"urls": ["https://example.com"]},
            )
            # Accept success or validation error (means service is working)
            assert response.status_code in [200, 422], (
                f"Crawl4AI crawl failed: {response.status_code} - {response.text}"
            )


class TestMCPProtocol:
    """Tests for MCP protocol endpoints."""

    def test_searxng_mcp_protocol(self):
        """Test SearXNG MCP responds to MCP-style requests."""
        with httpx.Client(timeout=TIMEOUT) as client:
            # Test MCP endpoint (not SSE - this server uses /mcp)
            response = client.get(f"http://{BASE_HOST}:38081/mcp")
            # 400 means endpoint exists but needs proper MCP request
            assert response.status_code in [200, 307, 400, 405], (
                f"SearXNG MCP endpoint failed: {response.status_code}"
            )

    def test_metamcp_sse_endpoint(self):
        """Test MetaMCP SSE endpoint for MCP connections."""
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            response = client.get(f"http://{BASE_HOST}:12008/sse")
            print(f"\nMetaMCP SSE status: {response.status_code}")
            # Check response content type for SSE
            content_type = response.headers.get("content-type", "")
            print(f"Content-Type: {content_type}")

    def test_metamcp_mcp_endpoint(self):
        """Test MetaMCP MCP JSON-RPC endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            # Try a JSON-RPC style request
            response = client.post(
                f"http://{BASE_HOST}:12008/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {},
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )
            print(f"\nMetaMCP MCP endpoint: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.text[:200]}")

    def test_crawl4ai_mcp_endpoint(self):
        """Test Crawl4AI MCP endpoint."""
        with httpx.Client(timeout=TIMEOUT) as client:
            # Check for MCP-style endpoints
            for endpoint in ["/mcp", "/sse", "/v1/tools"]:
                response = client.get(f"http://{BASE_HOST}:11235{endpoint}")
                print(f"\nCrawl4AI {endpoint}: {response.status_code}")


def main():
    """Quick health check for all MCP services."""
    import sys

    print("MCP Services Health Check")
    print("=" * 50)

    services = [
        ("SearXNG", f"http://{BASE_HOST}:38080", "/healthz"),
        ("SearXNG-MCP", f"http://{BASE_HOST}:38081", "/health"),
        ("Crawl4AI", f"http://{BASE_HOST}:11235", "/health"),
        ("MetaMCP", f"http://{BASE_HOST}:12008", "/health"),
    ]

    all_passed = True
    with httpx.Client(timeout=TIMEOUT) as client:
        for name, base_url, health_path in services:
            port = base_url.split(":")[-1]
            try:
                response = client.get(f"{base_url}{health_path}")
                if response.status_code == 200:
                    print(f"✓ {name:15} OK (:{port})")
                else:
                    print(f"⚠ {name:15} HTTP {response.status_code} (:{port})")
                    all_passed = False
            except httpx.RequestError as e:
                print(f"✗ {name:15} FAILED (:{port}) - {e}")
                all_passed = False

    print("=" * 50)
    if all_passed:
        print("All services are healthy!")
        return 0
    else:
        print("Some services have issues.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
