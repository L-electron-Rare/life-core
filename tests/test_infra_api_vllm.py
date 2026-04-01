"""Tests for vLLM health check in infra_api."""
import os
import pytest
from unittest.mock import patch, AsyncMock
from httpx import Response

from life_core.infra_api import network_status


@pytest.mark.asyncio
async def test_network_status_includes_vllm_when_configured():
    """vLLM health check appears when VLLM_BASE_URL is set."""
    health_response = Response(200, json={})
    models_response = Response(200, json={
        "data": [
            {"id": "qwen-27b-awq"},
            {"id": "mascarade-stm32"},
        ]
    })

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[health_response, models_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    env = {"VLLM_BASE_URL": "http://localhost:11436", "OLLAMA_URL": "", "OLLAMA_REMOTE_URL": ""}
    with patch.dict(os.environ, env, clear=False), \
         patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        result = await network_status()

    assert "vllm_gpu" in result
    assert result["vllm_gpu"]["status"] == "up"
    assert "qwen-27b-awq" in result["vllm_gpu"]["models"]


@pytest.mark.asyncio
async def test_network_status_vllm_down():
    """vLLM health check reports down when unreachable."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    env = {"VLLM_BASE_URL": "http://localhost:11436", "OLLAMA_URL": "", "OLLAMA_REMOTE_URL": ""}
    with patch.dict(os.environ, env, clear=False), \
         patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        result = await network_status()

    assert "vllm_gpu" in result
    assert result["vllm_gpu"]["status"] == "down"
