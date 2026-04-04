"""Extended tests for stats_api — percentile computation, error rates, timeseries shape."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import life_core.stats_api as stats_module
from life_core.stats_api import stats_router, record_call, _compute_percentile


@pytest.fixture(autouse=True)
def reset_stats():
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


# ---------------------------------------------------------------------------
# _compute_percentile pure function
# ---------------------------------------------------------------------------


def test_compute_percentile_empty_returns_zero():
    assert _compute_percentile([], 50) == 0.0


def test_compute_percentile_single_value():
    assert _compute_percentile([100.0], 50) == 100.0


def test_compute_percentile_p50():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _compute_percentile(data, 50)
    assert result == 30.0


def test_compute_percentile_p99_returns_last():
    data = [1.0, 2.0, 3.0, 4.0, 100.0]
    result = _compute_percentile(data, 99)
    assert result == 100.0


def test_compute_percentile_p0_returns_first_sorted():
    data = [50.0, 10.0, 30.0]
    result = _compute_percentile(data, 0)
    assert result == 10.0  # min after sort


# ---------------------------------------------------------------------------
# record_call
# ---------------------------------------------------------------------------


def test_record_call_increments_count():
    record_call("ollama", "qwen3:4b", 100.0, True)
    assert stats_module._call_count == 1
    assert stats_module._error_count == 0


def test_record_call_increments_error_count_on_failure():
    record_call("ollama", "qwen3:4b", 100.0, False)
    assert stats_module._error_count == 1


def test_record_call_appends_latency():
    record_call("ollama", "qwen3:4b", 250.0, True)
    assert 250.0 in stats_module._last_latencies


def test_record_multiple_calls_accumulates():
    for i in range(5):
        record_call("p", "m", float(i * 100), i % 2 == 0)
    assert stats_module._call_count == 5
    assert stats_module._error_count == 2  # i=1,3 are failures


# ---------------------------------------------------------------------------
# /stats/timeseries endpoint
# ---------------------------------------------------------------------------


def test_timeseries_default_points(client):
    response = client.get("/stats/timeseries")
    assert response.status_code == 200
    data = response.json()
    assert len(data["series"]) == 20  # default points=20


def test_timeseries_custom_points(client):
    response = client.get("/stats/timeseries?points=10")
    data = response.json()
    assert len(data["series"]) == 10


def test_timeseries_series_has_required_fields(client):
    response = client.get("/stats/timeseries?points=1")
    data = response.json()
    point = data["series"][0]
    assert "time" in point
    assert "timestamp" in point
    assert "p50" in point
    assert "p99" in point
    assert "calls" in point
    assert "errors" in point


def test_timeseries_summary_has_required_fields(client):
    response = client.get("/stats/timeseries")
    data = response.json()
    summary = data["summary"]
    assert "total_calls" in summary
    assert "total_errors" in summary
    assert "p50_ms" in summary
    assert "p99_ms" in summary
    assert "error_rate" in summary


def test_timeseries_error_rate_computed_correctly(client):
    record_call("p", "m", 100.0, True)
    record_call("p", "m", 200.0, False)
    response = client.get("/stats/timeseries")
    data = response.json()
    # 1 error out of 2 calls = 50%
    assert data["summary"]["error_rate"] == pytest.approx(50.0, abs=0.1)


def test_timeseries_zero_calls_error_rate_is_zero(client):
    response = client.get("/stats/timeseries")
    data = response.json()
    assert data["summary"]["error_rate"] == 0.0


def test_timeseries_p50_is_nonzero_after_calls(client):
    record_call("p", "m", 300.0, True)
    record_call("p", "m", 500.0, True)
    response = client.get("/stats/timeseries")
    data = response.json()
    assert data["summary"]["p50_ms"] > 0
