"""Tests pour l'évaluation LiteLLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router.providers.litellm_provider import LiteLLMProvider
from life_core.router import Router
from life_core.router.providers.base import LLMResponse


def test_litellm_provider_initialization():
    """Vérifier que LiteLLMProvider peut être initialisé."""
    with patch("life_core.router.providers.litellm_provider.litellm"):
        provider = LiteLLMProvider(model_prefix="openai/")
        assert provider.provider_id == "litellm"
        assert provider.model_prefix == "openai/"


@pytest.mark.asyncio
async def test_litellm_provider_implements_interface():
    """Vérifier que LiteLLMProvider implémente correctement l'interface LLMProvider."""
    with patch("life_core.router.providers.litellm_provider.litellm") as mock_litellm:
        # Mock la réponse litellm
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        mock_litellm.completion.return_value = mock_response
        mock_litellm.model_list = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        
        provider = LiteLLMProvider()
        
        # Test send()
        response = await provider.send(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        
        # Vérifier que c'est un LLMResponse valide
        assert isinstance(response, LLMResponse)
        assert response.content == "test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "litellm"
        assert "input_tokens" in response.usage
        assert "output_tokens" in response.usage
        
        # Test health_check()
        health = await provider.health_check()
        assert health is True
        
        # Test list_models()
        models = await provider.list_models()
        assert isinstance(models, list)


@pytest.mark.asyncio
async def test_litellm_provider_in_router():
    """Vérifier que LiteLLMProvider peut être enregistré dans le routeur sans modification."""
    with patch("life_core.router.providers.litellm_provider.litellm"):
        router = Router()
        
        # Créer provider LiteLLM
        litellm_provider = LiteLLMProvider(model_prefix="openai/")
        
        # L'enregistrer — aucune modification du routeur
        router.register_provider(litellm_provider, is_primary=False)
        
        # Vérifier qu'il est bien enregistré
        assert "litellm" in router.providers
        assert router.providers["litellm"] == litellm_provider


@pytest.mark.asyncio
async def test_litellm_call_with_fallback():
    """Vérifier que le routeur peut utiliser LiteLLM en fallback."""
    with patch("life_core.router.providers.litellm_provider.litellm") as mock_litellm:
        # Mock la réponse
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "litellm response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=15)
        
        mock_litellm.completion.return_value = mock_response
        
        router = Router()
        litellm_provider = LiteLLMProvider()
        
        router.register_provider(litellm_provider, is_primary=True)
        
        # Appeler via le routeur
        response = await router.send(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo"
        )
        
        assert response.content == "litellm response"
        assert response.provider == "litellm"
        assert response.usage["input_tokens"] == 5
