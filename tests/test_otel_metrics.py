"""Tests for OTEL metrics on cache and LLM operations."""
import pytest
from unittest.mock import patch, MagicMock

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry import metrics as otel_metrics

from life_core.cache.multi_tier_cache import MultiTierCache


@pytest.mark.asyncio
async def test_cache_hit_increments_counter():
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])

    test_meter = mp.get_meter("test")

    with patch("life_core.telemetry.get_meter", return_value=test_meter):
        cache = MultiTierCache()
        await cache.set("k", "v", ttl=60)
        await cache.get("k")

    metrics_data = reader.get_metrics_data()
    metric_names = []
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                metric_names.append(metric.name)

    assert "cache.hits" in metric_names


@pytest.mark.asyncio
async def test_cache_miss_increments_counter():
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    test_meter = mp.get_meter("test")

    with patch("life_core.telemetry.get_meter", return_value=test_meter):
        cache = MultiTierCache()
        await cache.get("nonexistent")

    metrics_data = reader.get_metrics_data()
    metric_names = []
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                metric_names.append(metric.name)

    assert "cache.misses" in metric_names
