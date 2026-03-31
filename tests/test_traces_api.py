"""Tests for traces API proxy."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from life_core.traces_api import traces_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(traces_router)
    return TestClient(app)


def test_services_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": ["life-core"]}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        response = client.get("/traces/services")
    assert response.status_code == 200


def test_services_failure(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("fail")):
        response = client.get("/traces/services")
    assert response.status_code == 200
    assert "error" in response.json()


def test_recent_traces(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        response = client.get("/traces/recent?service=life-core&limit=5")
    assert response.status_code == 200


def test_recent_traces_default_params(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        response = client.get("/traces/recent")
    assert response.status_code == 200


def test_recent_traces_failure(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("jaeger down")):
        response = client.get("/traces/recent?service=life-core")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["data"] == []


def test_services_returns_data(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": ["life-core", "life-reborn"]}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        response = client.get("/traces/services")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 2
