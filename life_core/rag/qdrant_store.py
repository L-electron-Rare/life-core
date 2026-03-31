"""VectorStore backed par Qdrant."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models

from .pipeline import Chunk

logger = logging.getLogger("life_core.rag.qdrant")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


class QdrantVectorStore:
    """Stockage de vecteurs via Qdrant."""

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "life_chunks"):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creer la collection si elle n'existe pas."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add(self, chunk_id: str, embedding: list[float], chunk: Chunk) -> None:
        """Ajouter un vecteur."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    },
                )
            ],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Rechercher les chunks les plus similaires."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        )
        chunks = []
        for point in results.points:
            payload = point.payload
            chunks.append(
                Chunk(
                    content=payload["content"],
                    document_id=payload["document_id"],
                    chunk_index=payload["chunk_index"],
                    metadata=payload.get("metadata", {}),
                )
            )
        return chunks
