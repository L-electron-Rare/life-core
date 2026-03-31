"""Tests pour le provider Ollama."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from life_core.router.providers.ollama import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider(base_url="http://localhost:11434", name="ollama-local")


@pytest.mark.asyncio
async def test_ollama_provider_name(provider):
    assert provider.provider_id == "ollama-local"


@pytest.mark.asyncio
async def test_ollama_send_formats_request(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Hello!"},
        "model": "llama3.2",
        "eval_count": 10,
        "prompt_eval_count": 5,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.send(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama3.2",
        )

    assert result.content == "Hello!"
    assert result.model == "llama3.2"
    assert result.provider == "ollama-local"
    assert result.usage["output_tokens"] == 10
    assert result.usage["input_tokens"] == 5


@pytest.mark.asyncio
async def test_ollama_list_models(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3.2:latest"},
            {"name": "mistral:7b"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        models = await provider.list_models()

    assert models == ["llama3.2:latest", "mistral:7b"]


@pytest.mark.asyncio
async def test_ollama_health_check_ok(provider):
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        healthy = await provider.health_check()

    assert healthy is True


@pytest.mark.asyncio
async def test_ollama_health_check_fail(provider):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
        healthy = await provider.health_check()

    assert healthy is False
