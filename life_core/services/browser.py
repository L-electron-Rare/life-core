"""Camoufox browser service used by scraping endpoints."""

from __future__ import annotations

from urllib.parse import urlparse


class BrowserServiceError(RuntimeError):
    """Base error for browser service failures."""


class BrowserDependencyMissingError(BrowserServiceError):
    """Raised when camoufox package is not installed."""


class BrowserService:
    """Minimal browser wrapper for Camoufox-powered page extraction."""

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BrowserServiceError("url must be a valid http(s) URL")

    async def scrape(
        self,
        *,
        url: str,
        selector: str | None = None,
        timeout_ms: int = 15000,
    ) -> dict[str, str]:
        self._validate_url(url)
        if timeout_ms <= 0:
            raise BrowserServiceError("timeout_ms must be > 0")

        return await self._run_camoufox(url=url, selector=selector, timeout_ms=timeout_ms)

    async def _run_camoufox(
        self,
        *,
        url: str,
        selector: str | None,
        timeout_ms: int,
    ) -> dict[str, str]:
        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError as exc:
            raise BrowserDependencyMissingError(
                "camoufox is not installed in life-core environment"
            ) from exc

        async with AsyncCamoufox() as browser:
            page = await browser.new_page()
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            title = await page.title()

            if selector:
                element = page.locator(selector).first
                extracted = await element.text_content()
                content = (extracted or "").strip()
            else:
                content = await page.content()

            return {
                "url": page.url,
                "title": title or "",
                "content": content or "",
            }
