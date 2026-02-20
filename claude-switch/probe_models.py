#!/usr/bin/env python3
"""Probe which Anthropic models work on a local proxy.

Usage:
    python claude-switch/probe_models.py
    python claude-switch/probe_models.py http://localhost:11212
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from urllib.request import Request, urlopen

import anthropic

PROXY_BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11212"
ANTHROPIC_URL = f"{PROXY_BASE}/api/anthropic"
TAGS_URL = f"{PROXY_BASE}/api/tags"
API_KEY = "test-key"


def fetch_anthropic_models() -> list[str]:
    """Get anthropic model names from /api/tags."""
    req = Request(TAGS_URL)
    with urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())

    models = []
    for m in data["models"]:
        family = m.get("details", {}).get("family", "")
        if family != "anthropic":
            continue
        name = m["name"]
        models.append(name)
        # Also try the short name (strip aws:anthropic. prefix)
        if name.startswith("aws:anthropic."):
            short = name.removeprefix("aws:anthropic.")
            # Strip version suffix like -v1:0
            for suffix in ("-v1:0", "-v2:0", "-v1"):
                if short.endswith(suffix):
                    short = short.removesuffix(suffix)
                    break
            if short != name:
                models.append(short)
    return sorted(set(models))


async def test_model(client: anthropic.AsyncAnthropic, model: str) -> tuple[str, bool, str, float]:
    """Test a single model. Returns (model, ok, detail, latency_ms)."""
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "say ok"}],
            ),
            timeout=10,
        )
        ms = (time.monotonic() - t0) * 1000
        text = resp.content[0].text if resp.content else ""
        return (model, True, f"→ {resp.model}  [{text.strip()[:20]}]", ms)
    except asyncio.TimeoutError:
        ms = (time.monotonic() - t0) * 1000
        return (model, False, "TIMEOUT", ms)
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        msg = str(e).split("\n")[0][:60]
        return (model, False, msg, ms)


async def main() -> None:
    print(f"Fetching models from {TAGS_URL} ...")
    models = fetch_anthropic_models()
    print(f"Found {len(models)} candidate model names\n")

    client = anthropic.AsyncAnthropic(
        base_url=ANTHROPIC_URL,
        api_key=API_KEY,
    )

    t0 = time.monotonic()
    results = await asyncio.gather(*(test_model(client, m) for m in models))
    elapsed = time.monotonic() - t0

    await client.close()

    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]

    print(f"{'MODEL':<45} {'LATENCY':>8}  DETAIL")
    print("─" * 90)
    for model, _, detail, ms in sorted(ok, key=lambda r: r[0]):
        print(f"  {model:<43} {ms:>6.0f}ms  {detail}")

    if fail:
        print()
        for model, _, detail, ms in sorted(fail, key=lambda r: r[0]):
            print(f"✗ {model:<43} {ms:>6.0f}ms  {detail}")

    print(f"\n{len(ok)} working, {len(fail)} failed — tested in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
