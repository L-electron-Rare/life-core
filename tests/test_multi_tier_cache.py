"""Tests for MultiTierCache, L1Cache, and L2Cache."""
from __future__ import annotations

import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.cache.multi_tier_cache import CacheEntry, L1Cache, L2Cache, MultiTierCache


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


def test_cache_entry_no_ttl_never_expires():
    entry = CacheEntry("k", "v", ttl=None)
    assert entry.is_expired() is False


def test_cache_entry_with_ttl_not_yet_expired():
    entry = CacheEntry("k", "v", ttl=60)
    assert entry.is_expired() is False


def test_cache_entry_expired():
    entry = CacheEntry("k", "v", ttl=1)
    entry.created_at = time.time() - 10  # simulate past creation
    assert entry.is_expired() is True


def test_cache_entry_hit_increments():
    entry = CacheEntry("k", "v")
    entry.hit()
    entry.hit()
    assert entry.hits == 2


# ---------------------------------------------------------------------------
# L1Cache — basic operations
# ---------------------------------------------------------------------------


def test_l1_set_and_get():
    cache = L1Cache()
    cache.set("foo", "bar")
    assert cache.get("foo") == "bar"


def test_l1_get_missing_returns_none():
    cache = L1Cache()
    assert cache.get("missing") is None


def test_l1_get_increments_hit_stat():
    cache = L1Cache()
    cache.set("foo", 42)
    cache.get("foo")
    assert cache.stats["hits"] == 1


def test_l1_miss_increments_miss_stat():
    cache = L1Cache()
    cache.get("nope")
    assert cache.stats["misses"] == 1


def test_l1_delete_removes_entry():
    cache = L1Cache()
    cache.set("foo", "bar")
    cache.delete("foo")
    assert cache.get("foo") is None


def test_l1_delete_nonexistent_is_noop():
    cache = L1Cache()
    cache.delete("nonexistent")  # must not raise


def test_l1_clear_empties_all_entries():
    cache = L1Cache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache.entries) == 0


def test_l1_get_stats_keys():
    cache = L1Cache(max_size=500)
    stats = cache.get_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "size" in stats
    assert stats["max_size"] == 500


# ---------------------------------------------------------------------------
# L1Cache — expired entry eviction (lines 68-69)
# ---------------------------------------------------------------------------


def test_l1_expired_entry_returns_none_and_is_deleted():
    cache = L1Cache()
    cache.set("exp", "value", ttl=1)
    # Backdate the creation time so the entry is already expired
    cache.entries["exp"].created_at = time.time() - 100
    result = cache.get("exp")
    assert result is None
    assert "exp" not in cache.entries


def test_l1_expired_entry_increments_miss_stat():
    cache = L1Cache()
    cache.set("exp", "value", ttl=1)
    cache.entries["exp"].created_at = time.time() - 100
    cache.get("exp")
    assert cache.stats["misses"] == 1


# ---------------------------------------------------------------------------
# L1Cache — LRU eviction (lines 76-80)
# ---------------------------------------------------------------------------


def test_l1_lru_eviction_removes_least_hit_entry():
    cache = L1Cache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # Give "a" and "c" more hits than "b"
    cache.get("a")
    cache.get("a")
    cache.get("c")
    # "b" has 0 hits → should be evicted when we add "d"
    cache.set("d", 4)
    assert "b" not in cache.entries
    assert "a" in cache.entries
    assert "c" in cache.entries
    assert "d" in cache.entries


def test_l1_lru_eviction_stays_within_max_size():
    cache = L1Cache(max_size=2)
    cache.set("x", 1)
    cache.set("y", 2)
    cache.set("z", 3)
    assert len(cache.entries) == 2


# ---------------------------------------------------------------------------
# L2Cache — no redis_url → unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_no_url_get_returns_none():
    cache = L2Cache(redis_url=None)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_l2_no_url_set_is_noop():
    cache = L2Cache(redis_url=None)
    await cache.set("k", "v")  # must not raise


@pytest.mark.asyncio
async def test_l2_no_url_delete_is_noop():
    cache = L2Cache(redis_url=None)
    await cache.delete("k")  # must not raise


@pytest.mark.asyncio
async def test_l2_no_url_clear_is_noop():
    cache = L2Cache(redis_url=None)
    await cache.clear()  # must not raise


def test_l2_no_url_stats():
    cache = L2Cache(redis_url=None)
    stats = cache.get_stats()
    assert stats["available"] is False


# ---------------------------------------------------------------------------
# L2Cache — Redis mocked: successful get
# ---------------------------------------------------------------------------


def _make_l2_with_mock_client(mock_client):
    """Return an L2Cache whose _client is already set to mock_client."""
    cache = L2Cache(redis_url="redis://localhost:6379/0")
    cache._client = mock_client
    return cache


@pytest.mark.asyncio
async def test_l2_get_hit_returns_json_decoded_value():
    client = AsyncMock()
    client.get = AsyncMock(return_value=json.dumps({"x": 1}).encode())
    cache = _make_l2_with_mock_client(client)

    result = await cache.get("key")
    assert result == {"x": 1}
    assert cache.stats["hits"] == 1


@pytest.mark.asyncio
async def test_l2_get_miss_returns_none():
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    cache = _make_l2_with_mock_client(client)

    result = await cache.get("key")
    assert result is None
    assert cache.stats["misses"] == 1


# ---------------------------------------------------------------------------
# L2Cache — JSON decode fallback (lines 145-151): corrupted JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_get_corrupted_json_returns_raw_value():
    client = AsyncMock()
    corrupted = b"not-valid-json{{"
    client.get = AsyncMock(return_value=corrupted)
    cache = _make_l2_with_mock_client(client)

    result = await cache.get("key")
    # Must not raise; returns raw bytes instead
    assert result == corrupted
    assert cache.stats["hits"] == 1


# ---------------------------------------------------------------------------
# L2Cache — connection error path (lines 127-129)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_get_client_connection_failure_marks_unavailable():
    """If redis.from_url raises, available becomes False."""
    cache = L2Cache(redis_url="redis://bad-host:9999/0")
    # _client is None → _get_client will be called

    fake_redis = MagicMock()
    fake_redis.from_url = AsyncMock(side_effect=ConnectionError("refused"))

    with patch.dict("sys.modules", {"redis": MagicMock(), "redis.asyncio": fake_redis}):
        result = await cache.get("k")

    assert result is None
    assert cache.available is False


@pytest.mark.asyncio
async def test_l2_get_raises_exception_returns_none():
    """If client.get raises, get() returns None without propagating."""
    client = AsyncMock()
    client.get = AsyncMock(side_effect=RuntimeError("boom"))
    cache = _make_l2_with_mock_client(client)

    result = await cache.get("key")
    assert result is None
    assert cache.stats["misses"] == 1


# ---------------------------------------------------------------------------
# L2Cache — set operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_set_without_ttl_calls_set():
    client = AsyncMock()
    client.set = AsyncMock()
    cache = _make_l2_with_mock_client(client)

    await cache.set("k", {"data": True})
    client.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_l2_set_with_ttl_calls_setex():
    client = AsyncMock()
    client.setex = AsyncMock()
    cache = _make_l2_with_mock_client(client)

    await cache.set("k", "val", ttl=30)
    client.setex.assert_awaited_once_with("k", 30, json.dumps("val"))


@pytest.mark.asyncio
async def test_l2_set_non_serializable_falls_back_to_str():
    client = AsyncMock()
    client.set = AsyncMock()
    cache = _make_l2_with_mock_client(client)

    class Unserializable:
        def __repr__(self):
            return "Unserializable()"

    obj = Unserializable()
    await cache.set("k", obj)
    # Should use str(obj), not raise
    client.set.assert_awaited_once()
    args = client.set.call_args[0]
    assert args[1] == str(obj)


@pytest.mark.asyncio
async def test_l2_set_error_does_not_propagate():
    client = AsyncMock()
    client.set = AsyncMock(side_effect=RuntimeError("redis down"))
    cache = _make_l2_with_mock_client(client)

    await cache.set("k", "v")  # must not raise


@pytest.mark.asyncio
async def test_l2_set_client_none_returns_early():
    """set() returns silently when _get_client returns None (line 168)."""
    cache = L2Cache(redis_url="redis://localhost:6379/0")
    # _client stays None; patch _get_client to always return None
    cache._get_client = AsyncMock(return_value=None)
    await cache.set("k", "v")  # must not raise, must not call set/setex


# ---------------------------------------------------------------------------
# L2Cache — delete / clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_delete_calls_client_delete():
    client = AsyncMock()
    client.delete = AsyncMock()
    cache = _make_l2_with_mock_client(client)

    await cache.delete("k")
    client.delete.assert_awaited_once_with("k")


@pytest.mark.asyncio
async def test_l2_delete_error_does_not_propagate():
    client = AsyncMock()
    client.delete = AsyncMock(side_effect=RuntimeError("down"))
    cache = _make_l2_with_mock_client(client)

    await cache.delete("k")  # must not raise


@pytest.mark.asyncio
async def test_l2_clear_calls_flushdb():
    client = AsyncMock()
    client.flushdb = AsyncMock()
    cache = _make_l2_with_mock_client(client)

    await cache.clear()
    client.flushdb.assert_awaited_once()


@pytest.mark.asyncio
async def test_l2_clear_error_does_not_propagate():
    client = AsyncMock()
    client.flushdb = AsyncMock(side_effect=RuntimeError("down"))
    cache = _make_l2_with_mock_client(client)

    await cache.clear()  # must not raise


# ---------------------------------------------------------------------------
# MultiTierCache — helpers
# ---------------------------------------------------------------------------


def _make_multi_tier_cache() -> MultiTierCache:
    """Return a MultiTierCache with L2 disabled (no Redis URL)."""
    with patch("life_core.telemetry.get_meter") as mock_get_meter:
        meter = MagicMock()
        meter.create_counter = MagicMock(return_value=MagicMock())
        mock_get_meter.return_value = meter
        cache = MultiTierCache(redis_url=None)
    return cache


# ---------------------------------------------------------------------------
# MultiTierCache — basic set/get/delete/clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multitier_set_and_get_l1_hit():
    cache = _make_multi_tier_cache()
    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        await cache.set("key", "value", ttl=60)
        result = await cache.get("key")
    assert result == "value"


@pytest.mark.asyncio
async def test_multitier_get_missing_returns_default():
    cache = _make_multi_tier_cache()
    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        result = await cache.get("missing", default="fallback")
    assert result == "fallback"


@pytest.mark.asyncio
async def test_multitier_delete_removes_from_l1():
    cache = _make_multi_tier_cache()
    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        await cache.set("k", "v")
        await cache.delete("k")
        result = await cache.get("k")
    assert result is None


@pytest.mark.asyncio
async def test_multitier_clear_empties_l1():
    cache = _make_multi_tier_cache()
    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.clear()
        assert len(cache.l1.entries) == 0


@pytest.mark.asyncio
async def test_multitier_get_stats_has_l1_and_l2():
    cache = _make_multi_tier_cache()
    stats = cache.get_stats()
    assert "l1" in stats
    assert "l2" in stats


@pytest.mark.asyncio
async def test_multitier_health_check():
    cache = _make_multi_tier_cache()
    health = await cache.health_check()
    assert health["l1"] is True
    assert health["l2"] is False  # no Redis


# ---------------------------------------------------------------------------
# MultiTierCache — L2 fallback + L1 backfill (lines 254-262)
# ---------------------------------------------------------------------------


def _setup_noop_tracer(mock_get_tracer):
    """Configure a tracer mock whose spans are context-manager compatible."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)

    tracer = MagicMock()
    tracer.start_as_current_span = MagicMock(return_value=span)
    mock_get_tracer.return_value = tracer


@pytest.mark.asyncio
async def test_multitier_l2_fallback_backfills_l1():
    """L1 miss → L2 hit → value backfilled into L1."""
    cache = _make_multi_tier_cache()

    # Pre-populate L2 mock; L1 is empty
    cache.l2.available = True
    cache.l2._client = AsyncMock()
    cache.l2._client.get = AsyncMock(return_value=json.dumps("from_redis").encode())

    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        result = await cache.get("redis_key")

    assert result == "from_redis"
    # L1 must have been backfilled
    assert cache.l1.get("redis_key") == "from_redis"


@pytest.mark.asyncio
async def test_multitier_l2_fallback_not_triggered_on_l1_hit():
    """L1 hit → L2.get must NOT be called."""
    cache = _make_multi_tier_cache()

    l2_get = AsyncMock(return_value=None)
    cache.l2.get = l2_get

    with patch("life_core.telemetry.get_tracer") as mock_tracer:
        _setup_noop_tracer(mock_tracer)
        await cache.set("k", "v")
        result = await cache.get("k")

    l2_get.assert_not_awaited()
    assert result == "v"
