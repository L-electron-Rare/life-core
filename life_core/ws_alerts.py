# life-core/life_core/ws_alerts.py
"""WebSocket endpoint for real-time infrastructure alerts."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("life_core.ws_alerts")

ws_router = APIRouter(tags=["Alerts"])

_POLL_INTERVAL = 15  # seconds

# --- Alert detection ---


def _check_gpu(gpu: dict[str, Any]) -> list[dict[str, Any]]:
    alerts = []
    if gpu.get("error"):
        alerts.append({"severity": "critical", "title": "GPU inference down",
                        "message": "vLLM metrics endpoint unreachable", "source": "gpu"})
    elif gpu.get("kv_cache_usage_percent", 0) > 95:
        pct = gpu["kv_cache_usage_percent"]
        alerts.append({"severity": "critical", "title": "VRAM critical",
                        "message": f"KV cache usage at {pct:.1f}% (>95% threshold)", "source": "gpu"})
    return alerts


def _check_containers(containers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    core = {"life-core", "life-reborn", "redis", "traefik"}
    for c in containers:
        if c.get("error"):
            alerts.append({"severity": "warning", "title": "Container stats unavailable",
                            "message": "Docker API unreachable on Tower", "source": "containers"})
            break
        if c["name"] in core and c.get("health") == "unhealthy":
            alerts.append({"severity": "warning", "title": f"Container degraded: {c['name']}",
                            "message": f"{c['name']} health check failing", "source": "containers"})
    return alerts


def _check_machines(machines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for m in machines:
        if m.get("cpu_percent", 0) > 90:
            alerts.append({"severity": "warning", "title": f"CPU overload: {m['name']}",
                            "message": f"{m['name']} CPU at {m['cpu_percent']}%", "source": "machines"})
    return alerts


def _check_flows(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for f in flows:
        if f.get("last_run_status") == "FAILED":
            alerts.append({"severity": "warning", "title": f"Flow failed: {f['name']}",
                            "message": f"Last run of '{f['name']}' failed", "source": "activepieces"})
    return alerts


async def _collect_alerts() -> list[dict[str, Any]]:
    """Poll all infra endpoints and return detected alerts."""
    from life_core.monitoring_api import gpu_stats, list_machines, activepieces_flows
    from life_core.infra_api import list_containers

    alerts: list[dict[str, Any]] = []
    try:
        gpu = await gpu_stats()
        alerts.extend(_check_gpu(gpu))
    except Exception as e:
        logger.debug("GPU check error: %s", e)

    try:
        machines_resp = await list_machines()
        alerts.extend(_check_machines(machines_resp.get("machines", [])))
    except Exception as e:
        logger.debug("Machines check error: %s", e)

    try:
        containers_resp = await list_containers()
        alerts.extend(_check_containers(containers_resp.get("containers", [])))
    except Exception as e:
        logger.debug("Containers check error: %s", e)

    try:
        flows_resp = await activepieces_flows()
        alerts.extend(_check_flows(flows_resp.get("flows", [])))
    except Exception as e:
        logger.debug("Flows check error: %s", e)

    # Attach ID and timestamp
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i, a in enumerate(alerts):
        a["id"] = f"alert-{int(time.time() * 1000)}-{i}"
        a["timestamp"] = ts

    return alerts


@ws_router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """Push infrastructure alerts to connected client."""
    await websocket.accept()
    logger.info("WebSocket client connected for alerts")
    try:
        while True:
            try:
                alerts = await _collect_alerts()
                for alert in alerts:
                    await websocket.send_json(alert)
            except Exception as exc:
                logger.warning("Alert collection error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
