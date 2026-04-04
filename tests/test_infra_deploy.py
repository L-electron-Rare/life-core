"""Tests for /infra/deploy endpoint."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from life_core.infra_api import infra_router


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(infra_router)
    return a


@pytest.fixture
def mock_docker():
    with patch("life_core.infra_api.docker.from_env") as m:
        client = MagicMock()
        m.return_value = client
        container = MagicMock()
        container.image.tags = ["ghcr.io/l-electron-rare/life-core:old"]
        container.attrs = {
            "Config": {"Env": []},
            "HostConfig": {"PortBindings": {}},
            "NetworkSettings": {"Networks": {"bridge": {}}},
        }
        client.containers.list.return_value = [container]
        yield client


@pytest.mark.asyncio
async def test_deploy_requires_token(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/infra/deploy",
            json={"service": "life-core", "image": "ghcr.io/x/life-core:latest"},
            headers={"X-Deploy-Token": "wrong"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_deploy_missing_token(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/infra/deploy",
            json={"service": "life-core", "image": "ghcr.io/x/life-core:latest"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_deploy_container_not_found(app, mock_docker):
    mock_docker.containers.list.return_value = []
    os.environ["DEPLOY_TOKEN"] = "testtoken"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/infra/deploy",
            json={"service": "life-core", "image": "ghcr.io/x/life-core:latest"},
            headers={"X-Deploy-Token": "testtoken"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_deploy_success(app, mock_docker):
    os.environ["DEPLOY_TOKEN"] = "testtoken"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/infra/deploy",
            json={"service": "life-core", "image": "ghcr.io/x/life-core:latest"},
            headers={"X-Deploy-Token": "testtoken"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "deployed"
    assert body["service"] == "life-core"
    mock_docker.images.pull.assert_called_once_with("ghcr.io/x/life-core:latest")
    mock_docker.containers.list.assert_called_once_with(filters={"name": "life-core"})
