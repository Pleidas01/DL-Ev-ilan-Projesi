"""Standalone Emlakjet detail-page HTML dumper for test fixtures.

Usage:
    python -m scraper._inspect --url <listing-url>

Default output: tests/fixtures/<listing_id>.html (idempotent — skips if exists).
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from playwright.async_api import async_playwright

from scraper.playwright_scraper import wait_for_cloudflare


async def fetch_html(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60_000)
            await wait_for_cloudflare(page)
            try:
                await page.get_by_text("Daha Fazla Gör").first.click(timeout=2000)
            except Exception:
                pass
            html = await page.content()
        finally:
            await context.close()
            await browser.close()
        return html


def listing_id_from_url(url: str) -> str:
    match = re.search(r"-(\d+)(?:\?|/|$)", url)
    if not match:
        raise ValueError(f"Could not extract listing id from URL: {url}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Emlakjet detail HTML for test fixtures")
    parser.add_argument("--url", required=True, help="Emlakjet detail URL")
    parser.add_argument("--out", default=None, help="Output file (default: tests/fixtures/<id>.html)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output exists")
    args = parser.parse_args()

    listing_id = listing_id_from_url(args.url)
    out_path = Path(args.out) if args.out else Path("tests/fixtures") / f"{listing_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"Exists, skipping: {out_path}")
        return

    html = asyncio.run(fetch_html(args.url))
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved {len(html)} chars -> {out_path}")


if __name__ == "__main__":
    main()
