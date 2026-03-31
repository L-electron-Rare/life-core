"""Infrastructure monitoring API endpoints."""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter

logger = logging.getLogger("life_core.infra_api")

infra_router = APIRouter(prefix="/infra", tags=["Infra"])


@infra_router.get("/containers")
async def list_containers():
    """List Docker containers via Tower's Docker socket proxy or SSH."""
    # For now, return static data from environment
    # Future: use Docker SDK or proxy
    return {
        "containers": [
            {"name": "life-core", "status": "healthy", "cpu": "~0.05%", "memory": "~92 MiB"},
            {"name": "life-reborn", "status": "healthy", "cpu": "~0.01%", "memory": "~21 MiB"},
            {"name": "life-web", "status": "healthy", "cpu": "~0.00%", "memory": "~13 MiB"},
            {"name": "redis", "status": "healthy", "cpu": "~0.28%", "memory": "~5 MiB"},
            {"name": "qdrant", "status": "running", "cpu": "~0.05%", "memory": "~58 MiB"},
            {"name": "forgejo", "status": "running", "cpu": "~0.07%", "memory": "~98 MiB"},
            {"name": "langfuse", "status": "running", "cpu": "~0.10%", "memory": "~150 MiB"},
            {"name": "jaeger", "status": "running", "cpu": "~0.02%", "memory": "~30 MiB"},
            {"name": "otel-collector", "status": "running", "cpu": "~0.01%", "memory": "~25 MiB"},
            {"name": "traefik", "status": "running", "cpu": "~0.00%", "memory": "~19 MiB"},
        ]
    }


@infra_router.get("/storage")
async def storage_stats():
    """Get storage stats from Redis and Qdrant."""
    result = {"redis": {}, "qdrant": {}}

    # Redis stats
    try:
        import redis as redis_lib
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        r = redis_lib.from_url(redis_url)
        info = r.info("memory")
        result["redis"] = {
            "status": "connected",
            "used_memory_human": info.get("used_memory_human", "?"),
            "connected_clients": r.info("clients").get("connected_clients", 0),
            "keys": r.dbsize(),
        }
        r.close()
    except Exception as e:
        result["redis"] = {"status": "error", "error": str(e)}

    # Qdrant stats
    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{qdrant_url}/collections")
            if resp.status_code == 200:
                data = resp.json()
                collections = data.get("result", {}).get("collections", [])
                result["qdrant"] = {
                    "status": "connected",
                    "collections": len(collections),
                    "collection_names": [c["name"] for c in collections],
                }
            else:
                result["qdrant"] = {"status": "error", "code": resp.status_code}
    except Exception as e:
        result["qdrant"] = {"status": "error", "error": str(e)}

    return result


@infra_router.get("/network")
async def network_status():
    """Check network connectivity to external services."""
    checks = {}

    # Ollama local
    ollama_url = os.environ.get("OLLAMA_URL", "")
    if ollama_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                models = resp.json().get("models", [])
                checks["ollama_local"] = {"status": "up", "models": len(models), "url": ollama_url}
        except Exception as e:
            checks["ollama_local"] = {"status": "down", "error": str(e), "url": ollama_url}

    # Ollama remote (KXKM-AI)
    ollama_remote = os.environ.get("OLLAMA_REMOTE_URL", "")
    if ollama_remote:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{ollama_remote}/api/tags")
                models = resp.json().get("models", [])
                checks["ollama_gpu"] = {"status": "up", "models": len(models), "url": ollama_remote}
        except Exception as e:
            checks["ollama_gpu"] = {"status": "down", "error": str(e), "url": ollama_remote}

    # Jaeger
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://jaeger:16686/api/services")
            checks["jaeger"] = {"status": "up" if resp.status_code == 200 else "down"}
    except Exception:
        checks["jaeger"] = {"status": "down"}

    return checks
