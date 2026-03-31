"""Tests for Cils docstore client."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from life_core.docstore_client import search_docstore, augment_with_docstore


@pytest.mark.asyncio
async def test_search_empty_url():
    with patch("life_core.docstore_client.DOCSTORE_URL", ""):
        results = await search_docstore("test")
    assert results == []


@pytest.mark.asyncio
async def test_search_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"content": "hello", "score": 0.8}]}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            results = await search_docstore("test")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_returns_empty_on_non_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            results = await search_docstore("test")
    assert results == []


@pytest.mark.asyncio
async def test_search_failure():
    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("timeout")):
            results = await search_docstore("test")
    assert results == []


@pytest.mark.asyncio
async def test_search_top_k_passed():
    """Verify top_k parameter is forwarded."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp) as mock_get:
            await search_docstore("hello", top_k=5)
    call_kwargs = mock_get.call_args
    assert call_kwargs is not None


@pytest.mark.asyncio
async def test_augment_filters_low_scores():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [
        {"content": "relevant", "document_name": "doc1", "score": 0.7},
        {"content": "irrelevant", "document_name": "doc2", "score": 0.2},
    ]}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            context = await augment_with_docstore("test")
    assert "relevant" in context
    assert "irrelevant" not in context


@pytest.mark.asyncio
async def test_augment_returns_empty_when_no_results():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            context = await augment_with_docstore("test")
    assert context == ""


@pytest.mark.asyncio
async def test_augment_returns_empty_when_url_not_set():
    with patch("life_core.docstore_client.DOCSTORE_URL", ""):
        context = await augment_with_docstore("test")
    assert context == ""


@pytest.mark.asyncio
async def test_augment_includes_document_name():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [
        {"content": "some content", "document_name": "manual.pdf", "score": 0.9},
    ]}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            context = await augment_with_docstore("test")
    assert "manual.pdf" in context
    assert "some content" in context


@pytest.mark.asyncio
async def test_augment_threshold_boundary():
    """Score exactly at 0.4 should be excluded (score > 0.4 required)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [
        {"content": "borderline", "document_name": "doc", "score": 0.4},
        {"content": "just above", "document_name": "doc2", "score": 0.41},
    ]}

    with patch("life_core.docstore_client.DOCSTORE_URL", "http://fake:8200"):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            context = await augment_with_docstore("test")
    assert "borderline" not in context
    assert "just above" in context
