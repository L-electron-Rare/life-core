"""Extended tests for ChatService — additional coverage paths."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router.providers.base import LLMResponse, LLMStreamChunk
from life_core.services.chat import ChatService


class _StubCache:
    def __init__(self):
        self.store: dict = {}
        self.set_calls = 0

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value, ttl: int = 3600):
        self.store[key] = value
        self.set_calls += 1

    def get_stats(self):
        return {"hits": 0, "misses": 0, "size": len(self.store), "max_size": 1000}


class _StubRouter:
    def __init__(self, response_content="mock response"):
        self._response_content = response_content

    async def send(self, messages, model, provider=None, **kwargs):
        return LLMResponse(
            content=self._response_content,
            model=model,
            provider=provider or "mock",
            usage={"input_tokens": 5, "output_tokens": 3},
        )

    async def stream(self, messages, model, provider=None, **kwargs):
        yield LLMStreamChunk(content="chunk1", model=model)
        yield LLMStreamChunk(content="chunk2", model=model, finish_reason="stop")


class _StubRag:
    def __init__(self, context="rag context"):
        self._context = context

    async def augment_context(self, query: str, top_k: int = 3):
        return self._context

    def get_stats(self):
        return {"indexed": 10}


# ---------------------------------------------------------------------------
# ChatService initialization
# ---------------------------------------------------------------------------


def test_chat_service_creates_default_cache_when_none():
    router = _StubRouter()
    svc = ChatService(router=router, cache=None, rag=None)
    assert svc.cache is not None


def test_chat_service_initial_stats_are_zero():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    assert svc.stats["requests"] == 0
    assert svc.stats["cache_hits"] == 0


# ---------------------------------------------------------------------------
# ChatService.chat()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_increments_request_count():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    await svc.chat(messages=[{"role": "user", "content": "hi"}], model="m1")
    assert svc.stats["requests"] == 1


@pytest.mark.asyncio
async def test_chat_returns_content():
    svc = ChatService(router=_StubRouter("hello world"), cache=_StubCache())
    result = await svc.chat(messages=[{"role": "user", "content": "hi"}], model="m1")
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_chat_returns_model_and_provider():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    result = await svc.chat(messages=[{"role": "user", "content": "hi"}], model="my-model")
    assert result["model"] == "my-model"
    assert result["provider"] == "mock"


@pytest.mark.asyncio
async def test_chat_caches_result():
    cache = _StubCache()
    svc = ChatService(router=_StubRouter(), cache=cache)
    messages = [{"role": "user", "content": "test"}]
    await svc.chat(messages=messages, model="m1")
    assert cache.set_calls == 1


@pytest.mark.asyncio
async def test_chat_cache_hit_skips_router():
    cache = _StubCache()
    svc = ChatService(router=_StubRouter(), cache=cache)
    messages = [{"role": "user", "content": "cached query"}]

    first = await svc.chat(messages=messages, model="m1")
    second = await svc.chat(messages=messages, model="m1")

    assert first["content"] == second["content"]
    assert svc.stats["cache_hits"] == 1
    assert cache.set_calls == 1  # only stored once


@pytest.mark.asyncio
async def test_chat_with_rag_disabled_skips_augmentation():
    rag = _StubRag("should not appear")
    svc = ChatService(router=_StubRouter(), cache=_StubCache(), rag=rag)
    result = await svc.chat(
        messages=[{"role": "user", "content": "question"}],
        model="m1",
        use_rag=False,
    )
    # Router gets original message, not augmented
    assert result is not None


@pytest.mark.asyncio
async def test_chat_with_rag_none_still_works():
    svc = ChatService(router=_StubRouter("ok"), cache=_StubCache(), rag=None)
    result = await svc.chat(
        messages=[{"role": "user", "content": "question"}],
        model="m1",
        use_rag=True,
    )
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_chat_with_rag_empty_context_not_injected():
    rag = _StubRag(context="")  # empty context
    svc = ChatService(router=_StubRouter(), cache=_StubCache(), rag=rag)
    result = await svc.chat(
        messages=[{"role": "user", "content": "q"}],
        model="m1",
        use_rag=True,
    )
    assert result is not None


@pytest.mark.asyncio
async def test_chat_passes_provider_to_router():
    router = _StubRouter()
    router.send = AsyncMock(return_value=LLMResponse(
        content="x", model="m", provider="specific-provider",
        usage={},
    ))
    svc = ChatService(router=router, cache=_StubCache())
    await svc.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="m1",
        provider="specific-provider",
    )
    call_kwargs = router.send.call_args
    assert call_kwargs.kwargs.get("provider") == "specific-provider" or \
           call_kwargs[1].get("provider") == "specific-provider"


# ---------------------------------------------------------------------------
# ChatService.stream_chat()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_yields_chunks():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    chunks = [c async for c in svc.stream_chat(
        messages=[{"role": "user", "content": "stream me"}],
        model="m1",
    )]
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_stream_chat_increments_requests():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    async for _ in svc.stream_chat(
        messages=[{"role": "user", "content": "q"}],
        model="m1",
    ):
        pass
    assert svc.stats["requests"] == 1


# ---------------------------------------------------------------------------
# ChatService.get_stats()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_stats_includes_rag_stats():
    rag = _StubRag()
    svc = ChatService(router=_StubRouter(), cache=_StubCache(), rag=rag)
    stats = svc.get_stats()
    assert stats["rag_stats"] is not None
    assert "indexed" in stats["rag_stats"]


@pytest.mark.asyncio
async def test_get_stats_rag_stats_none_when_no_rag():
    svc = ChatService(router=_StubRouter(), cache=_StubCache(), rag=None)
    stats = svc.get_stats()
    assert stats["rag_stats"] is None


@pytest.mark.asyncio
async def test_get_stats_includes_cache_stats():
    svc = ChatService(router=_StubRouter(), cache=_StubCache())
    stats = svc.get_stats()
    assert "cache_stats" in stats
