"""Tests for browser scraping PoC."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from life_core.services.browser import BrowserDependencyMissingError, BrowserService, BrowserServiceError


@pytest.mark.asyncio
async def test_browser_service_validates_url():
    service = BrowserService()

    with pytest.raises(BrowserServiceError):
        await service.scrape(url="file:///etc/passwd")


@pytest.mark.asyncio
async def test_browser_service_scrape_with_mock(monkeypatch):
    service = BrowserService()

    async def _fake_run_camoufox(*, url: str, selector: str | None, timeout_ms: int):
        assert url == "https://example.com"
        assert selector == "h1"
        assert timeout_ms == 5000
        return {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "Example Domain",
        }

    monkeypatch.setattr(service, "_run_camoufox", _fake_run_camoufox)

    result = await service.scrape(url="https://example.com", selector="h1", timeout_ms=5000)
    assert result["title"] == "Example Domain"


@pytest.mark.asyncio
async def test_scrape_endpoint_error_mapping(monkeypatch):
    import life_core.api as api

    class _StubBrowserService:
        async def scrape(self, **kwargs):
            raise BrowserDependencyMissingError("camoufox is missing")

    api.browser_service = _StubBrowserService()

    with pytest.raises(HTTPException) as err:
        await api.scrape(api.ScrapeRequest(url="https://example.com"))

    assert err.value.status_code == 503
