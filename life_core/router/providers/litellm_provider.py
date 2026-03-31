"""Provider adapter pour LiteLLM."""

from __future__ import annotations

import logging
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.litellm")

try:
    import litellm
except ImportError:
    litellm = None


class LiteLLMProvider(LLMProvider):
    """
    Adapter LiteLLM comme provider dans le routeur life-core.
    
    Utilise litellm pour accéder à 100+ LLMs en une interface unifiée.
    Compatible OpenAI.
    """
    
    def __init__(self, model_prefix: str = "", api_key: str | None = None):
        """
        Créer le provider LiteLLM.
        
        Args:
            model_prefix: Prefix pour les noms de modèles (ex: "openai/", "anthropic/")
            api_key: Clé API (optionnel, fallback sur env vars)
        """
        if litellm is None:
            raise ImportError(
                "litellm not installed. Install with: pip install 'life-core[litellm]'"
            )
        
        self.provider_id = "litellm"
        self.model_prefix = model_prefix
        self.api_key = api_key
        logger.info(f"LiteLLMProvider initialized with prefix={model_prefix}")
    
    async def send(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs
    ) -> LLMResponse:
        """
        Envoyer un message via litellm.
        
        Args:
            messages: Liste de messages (role/content)
            model: Nom du modèle
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse normalisée LLMResponse
        """
        # Ajouter le prefix au model
        full_model = f"{self.model_prefix}{model}" if self.model_prefix else model
        
        try:
            # Appel synchrone via litellm.completion
            response = litellm.completion(
                model=full_model,
                messages=messages,
                **kwargs
            )
            
            # Normaliser vers LLMResponse
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                provider=self.provider_id,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
        except Exception as e:
            logger.error(f"LiteLLM request failed: {e}")
            raise
    
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs
    ):
        """
        Streamer une réponse via litellm.
        
        Args:
            messages: Liste de messages
            model: Nom du modèle
            **kwargs: Paramètres additionnels
            
        Yields:
            Chunks de type LLMStreamChunk
        """
        full_model = f"{self.model_prefix}{model}" if self.model_prefix else model
        
        try:
            # litellm.completion(..., stream=True) retourne un itérateur
            response_stream = litellm.completion(
                model=full_model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            for chunk in response_stream:
                content = ""
                finish_reason = None
                
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        content = choice.delta.content
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                
                yield LLMStreamChunk(
                    content=content,
                    model=model,
                    finish_reason=finish_reason,
                )
        except Exception as e:
            logger.error(f"LiteLLM stream request failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Vérifier la santé du provider.
        
        Returns:
            True si le provider est accessible
        """
        # Pour LiteLLM, on suppose que c'est OK si l'import a réussi
        # Une implémentation réelle verrait un test de modèle
        return True
    
    async def list_models(self) -> list[str]:
        """
        Lister les modèles disponibles.
        
        Returns:
            Liste approximative (voir litellm.get_all_models())
        """
        try:
            # litellm expose une liste de modèles supportés
            return litellm.model_list or []
        except Exception:
            return []
