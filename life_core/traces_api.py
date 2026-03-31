"""Traces API — proxy to Jaeger for the cockpit."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

logger = logging.getLogger("life_core.traces_api")

traces_router = APIRouter(prefix="/traces", tags=["Traces"])

JAEGER_URL = "http://jaeger:16686"


@traces_router.get("/services")
async def list_services():
    """List traced services from Jaeger."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{JAEGER_URL}/api/services")
            return resp.json()
    except Exception as e:
        return {"data": [], "error": str(e)}


@traces_router.get("/recent")
async def recent_traces(service: str = "life-core", limit: int = 20):
    """Get recent traces from Jaeger."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{JAEGER_URL}/api/traces",
                params={"service": service, "limit": limit, "lookback": "1h"},
            )
            return resp.json()
    except Exception as e:
        return {"data": [], "error": str(e)}
