"""Tests for logs API."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import life_core.logs_api as logs_module
from life_core.logs_api import logs_router, add_log


@pytest.fixture(autouse=True)
def reset_logs():
    """Reset module-level log buffer before each test to avoid state pollution."""
    logs_module._log_buffer.clear()
    yield
    logs_module._log_buffer.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(logs_router)
    return TestClient(app)


def test_recent_logs_empty(client):
    response = client.get("/logs/recent?limit=10")
    assert response.status_code == 200
    assert response.json()["total"] >= 0


def test_recent_logs_with_entries(client):
    add_log("INFO", "test message 1")
    add_log("WARN", "test warning")
    add_log("ERROR", "test error")

    response = client.get("/logs/recent?limit=10")
    data = response.json()
    assert data["total"] >= 3
    assert any(l["level"] == "ERROR" for l in data["logs"])
