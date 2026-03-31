"""Tests for stats timeseries API."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import life_core.stats_api as stats_module
from life_core.stats_api import stats_router, record_call


@pytest.fixture(autouse=True)
def reset_stats():
    """Reset module-level globals before each test to avoid state pollution."""
    stats_module._metrics_buffer.clear()
    stats_module._last_latencies.clear()
    stats_module._call_count = 0
    stats_module._error_count = 0
    yield
    stats_module._metrics_buffer.clear()
    stats_module._last_latencies.clear()
    stats_module._call_count = 0
    stats_module._error_count = 0


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(stats_router)
    return TestClient(app)


def test_timeseries_empty(client):
    response = client.get("/stats/timeseries?points=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["series"]) == 5
    assert data["summary"]["total_calls"] == 0


def test_timeseries_after_calls(client):
    record_call("ollama", "qwen3:4b", 234.0, True)
    record_call("ollama", "qwen3:4b", 189.0, True)
    record_call("claude", "claude-3", 500.0, False)

    response = client.get("/stats/timeseries?points=3")
    data = response.json()
    assert data["summary"]["total_calls"] == 3
    assert data["summary"]["total_errors"] == 1
    assert data["summary"]["p50_ms"] > 0
