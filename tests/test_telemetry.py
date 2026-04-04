"""Tests pour la télémétrie OpenTelemetry."""

import importlib
import os
import sys
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_telemetry_module():
    """Reset module-level globals _tracer and _meter to None."""
    import life_core.telemetry as tel
    tel._tracer = None
    tel._meter = None


# ---------------------------------------------------------------------------
# init_telemetry() — endpoint NOT set → early return
# ---------------------------------------------------------------------------

def test_init_telemetry_no_endpoint_skips(caplog):
    """init_telemetry() doit logger et retourner sans rien configurer quand l'endpoint est absent."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    env_without_endpoint = {k: v for k, v in os.environ.items() if k != "OTEL_EXPORTER_OTLP_ENDPOINT"}
    with patch.dict(os.environ, env_without_endpoint, clear=True):
        with caplog.at_level("INFO", logger="life_core.telemetry"):
            tel.init_telemetry()

    assert tel._tracer is None
    assert tel._meter is None
    assert "OpenTelemetry disabled" in caplog.text


# ---------------------------------------------------------------------------
# init_telemetry() — endpoint set → full setup
# ---------------------------------------------------------------------------

def test_init_telemetry_with_endpoint_sets_providers(caplog):
    """init_telemetry() doit configurer _tracer et _meter quand l'endpoint est défini."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    mock_tracer = MagicMock(name="tracer")
    mock_meter = MagicMock(name="meter")
    mock_resource = MagicMock(name="resource")
    mock_tracer_provider = MagicMock(name="tracer_provider")
    mock_meter_provider = MagicMock(name="meter_provider")

    with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"}):
        with (
            patch("opentelemetry.sdk.resources.Resource.create", return_value=mock_resource),
            patch("opentelemetry.sdk.trace.TracerProvider", return_value=mock_tracer_provider),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", return_value=MagicMock()),
            patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter", return_value=MagicMock()),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
            patch("opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader", return_value=MagicMock()),
            patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter", return_value=MagicMock()),
            patch("opentelemetry.sdk.metrics.MeterProvider", return_value=mock_meter_provider),
            patch("opentelemetry.metrics.set_meter_provider"),
            patch("opentelemetry.metrics.get_meter", return_value=mock_meter),
        ):
            with caplog.at_level("INFO", logger="life_core.telemetry"):
                tel.init_telemetry()

    assert tel._tracer is mock_tracer
    assert tel._meter is mock_meter
    assert "OpenTelemetry initialized" in caplog.text


# ---------------------------------------------------------------------------
# init_telemetry() — ImportError path
# ---------------------------------------------------------------------------

def test_init_telemetry_import_error_logs_warning(caplog):
    """Quand les packages OTEL ne sont pas installés, init_telemetry doit logger un warning."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    # Simulate ImportError by removing the opentelemetry.sdk.trace key
    broken_modules = {
        "opentelemetry.sdk.trace": None,  # None causes ImportError in sys.modules
    }

    with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"}):
        with patch("builtins.__import__", side_effect=ImportError("No module named 'opentelemetry.sdk.trace'")):
            with caplog.at_level("WARNING", logger="life_core.telemetry"):
                tel.init_telemetry()

    assert tel._tracer is None
    assert "OpenTelemetry packages not installed" in caplog.text


# ---------------------------------------------------------------------------
# init_telemetry() — generic Exception path
# ---------------------------------------------------------------------------

def test_init_telemetry_generic_exception_logs_warning(caplog):
    """Quand une exception inattendue survient, init_telemetry doit logger un warning."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"}):
        with patch("builtins.__import__", side_effect=RuntimeError("unexpected failure")):
            with caplog.at_level("WARNING", logger="life_core.telemetry"):
                tel.init_telemetry()

    assert tel._tracer is None
    assert "OpenTelemetry init failed" in caplog.text


# ---------------------------------------------------------------------------
# get_tracer() — no init → returns no-op tracer
# ---------------------------------------------------------------------------

def test_get_tracer_without_init_returns_noop():
    """get_tracer() sans init doit retourner un tracer fonctionnel (no-op)."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    tracer = tel.get_tracer()
    assert tracer is not None
    # Must support start_as_current_span context manager
    with tracer.start_as_current_span("noop.span") as span:
        assert span is not None


# ---------------------------------------------------------------------------
# get_tracer() — after init with endpoint → returns configured tracer
# ---------------------------------------------------------------------------

def test_get_tracer_after_init_returns_configured_tracer():
    """get_tracer() doit retourner _tracer si init_telemetry() a réussi."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    sentinel = MagicMock(name="configured-tracer")
    tel._tracer = sentinel

    result = tel.get_tracer()
    assert result is sentinel


# ---------------------------------------------------------------------------
# get_meter() — no init → returns no-op meter
# ---------------------------------------------------------------------------

def test_get_meter_without_init_returns_noop():
    """get_meter() sans init doit retourner un meter fonctionnel (no-op)."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    meter = tel.get_meter()
    assert meter is not None


# ---------------------------------------------------------------------------
# get_meter() — after init with endpoint → returns configured meter
# ---------------------------------------------------------------------------

def test_get_meter_after_init_returns_configured_meter():
    """get_meter() doit retourner _meter si init_telemetry() a réussi."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    sentinel = MagicMock(name="configured-meter")
    tel._meter = sentinel

    result = tel.get_meter()
    assert result is sentinel


# ---------------------------------------------------------------------------
# init_telemetry() — OTEL_SERVICE_NAME override
# ---------------------------------------------------------------------------

def test_init_telemetry_custom_service_name():
    """init_telemetry() doit utiliser OTEL_SERVICE_NAME si défini."""
    import life_core.telemetry as tel
    _reset_telemetry_module()

    captured_resource_attrs = {}

    def fake_resource_create(attrs):
        captured_resource_attrs.update(attrs)
        return MagicMock()

    mock_tracer = MagicMock(name="tracer")
    mock_meter = MagicMock(name="meter")
    mock_tracer_provider = MagicMock(name="tracer_provider")
    mock_meter_provider = MagicMock(name="meter_provider")

    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317",
        "OTEL_SERVICE_NAME": "my-custom-service",
    }
    with patch.dict(os.environ, env):
        with (
            patch("opentelemetry.sdk.resources.Resource.create", side_effect=fake_resource_create),
            patch("opentelemetry.sdk.trace.TracerProvider", return_value=mock_tracer_provider),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", return_value=MagicMock()),
            patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter", return_value=MagicMock()),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
            patch("opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader", return_value=MagicMock()),
            patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter", return_value=MagicMock()),
            patch("opentelemetry.sdk.metrics.MeterProvider", return_value=mock_meter_provider),
            patch("opentelemetry.metrics.set_meter_provider"),
            patch("opentelemetry.metrics.get_meter", return_value=mock_meter),
        ):
            tel.init_telemetry()

    assert captured_resource_attrs.get("service.name") == "my-custom-service"
