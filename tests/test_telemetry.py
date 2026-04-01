"""Tests pour la télémétrie OpenTelemetry."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router import Router, LLMProvider
from life_core.router.providers.base import LLMResponse


@pytest.mark.asyncio
async def test_router_spans_emitted():
    """Vérifier que les spans sont émis lors d'un appel routeur."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Setup exporter en mémoire
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Patch get_tracer to use our test provider (set_tracer_provider is one-shot in OTEL)
    test_tracer = tracer_provider.get_tracer("life_core.router")

    # Créer un routeur avec un mock provider
    router = Router()

    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.provider_id = "test-provider"
    mock_provider.send = AsyncMock(return_value=LLMResponse(
        content="test response",
        model="test-model",
        provider="test-provider",
        usage={"input_tokens": 10, "output_tokens": 20}
    ))

    router.register_provider(mock_provider, is_primary=True)

    # Patch the module-level tracer in the router (resolved at import time)
    with patch("life_core.router.router.tracer", test_tracer):
        response = await router.send(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model"
        )

    # Vérifier les spans émis
    spans = exporter.get_finished_spans()
    
    # Devrait avoir au moins une span "llm.call"
    llm_call_spans = [s for s in spans if s.name == "llm.call"]
    assert len(llm_call_spans) > 0, f"Expected llm.call span, got: {[s.name for s in spans]}"
    
    # Vérifier les attributs de la span
    span = llm_call_spans[0]
    assert span.attributes.get("llm.model") == "test-model"
    assert span.attributes.get("llm.provider") == "test-provider"
    assert span.attributes.get("llm.tokens.input") == 10
    assert span.attributes.get("llm.tokens.output") == 20
    
    # Vérifier la réponse
    assert response.content == "test response"
    assert response.provider == "test-provider"


@pytest.mark.asyncio
async def test_tracer_initialization():
    """Vérifier que le tracer est bien initialisé."""
    from life_core.telemetry import get_tracer, init_telemetry
    
    init_telemetry()
    tracer = get_tracer()
    
    # Le tracer ne devrait pas être None
    assert tracer is not None
    
    # Pouvoir créer une span
    with tracer.start_as_current_span("test.span") as span:
        span.set_attribute("test.attribute", "value")
        assert span is not None
