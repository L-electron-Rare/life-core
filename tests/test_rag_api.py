"""Tests for RAG API endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from life_core.rag.api import rag_router, set_rag_pipeline, _get_rag
from life_core.rag.pipeline import RAGPipeline, Chunk


@pytest.fixture
def mock_rag():
    rag = MagicMock(spec=RAGPipeline)
    rag.get_stats.return_value = {"documents": 2, "chunks": 10, "vectors": 10}
    rag.query = AsyncMock(return_value=[
        Chunk(content="test result", document_id="doc1", chunk_index=0),
    ])
    rag.index_document = AsyncMock()
    set_rag_pipeline(rag)
    yield rag
    set_rag_pipeline(None)


def test_rag_stats(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == 2
    assert data["chunks"] == 10


def test_rag_search(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/search?q=test+query&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "test result"


def test_rag_stats_no_pipeline():
    from fastapi import FastAPI
    set_rag_pipeline(None)
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/stats")
    assert response.status_code == 503
