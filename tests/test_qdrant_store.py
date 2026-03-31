"""Tests pour le QdrantVectorStore."""

import pytest
from unittest.mock import MagicMock, patch

from life_core.rag.pipeline import Chunk


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    return client


@pytest.fixture
def store(mock_client):
    with patch("life_core.rag.qdrant_store.QdrantClient", return_value=mock_client):
        from life_core.rag.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(url="http://localhost:6333", collection_name="test_chunks")


def test_store_init(store):
    assert store.collection_name == "test_chunks"


def test_ensure_collection_called(store, mock_client):
    mock_client.create_collection.assert_called_once()


def test_add_vector(store, mock_client):
    chunk = Chunk(content="test content", document_id="doc1", chunk_index=0)
    embedding = [0.1, 0.2, 0.3]
    store.add("chunk_1", embedding, chunk)
    mock_client.upsert.assert_called_once()


def test_search(store, mock_client):
    query_embedding = [0.1, 0.2, 0.3]

    mock_point = MagicMock()
    mock_point.payload = {
        "content": "found content",
        "document_id": "doc1",
        "chunk_index": 0,
        "metadata": {},
    }
    mock_client.query_points.return_value.points = [mock_point]

    results = store.search(query_embedding, top_k=3)
    assert len(results) == 1
    assert results[0].content == "found content"
    assert results[0].document_id == "doc1"
