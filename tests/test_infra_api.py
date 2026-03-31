"""Tests for infra API."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from life_core.infra_api import infra_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(infra_router)
    return TestClient(app)


def test_containers(client):
    response = client.get("/infra/containers")
    assert response.status_code == 200
    containers = response.json()["containers"]
    assert len(containers) > 0
    assert any(c["name"] == "life-core" for c in containers)


def test_storage(client):
    response = client.get("/infra/storage")
    assert response.status_code == 200
    data = response.json()
    assert "redis" in data
    assert "qdrant" in data


def test_network(client):
    response = client.get("/infra/network")
    assert response.status_code == 200
