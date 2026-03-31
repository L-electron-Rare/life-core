"""Langfuse LLM tracing integration."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("life_core.langfuse")

_langfuse = None


def init_langfuse() -> None:
    """Initialize Langfuse if credentials are set."""
    global _langfuse

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        logger.info("Langfuse disabled (LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST not set)")
        return

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"Langfuse initialized, host={host}")
    except ImportError:
        logger.warning("langfuse package not installed")
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")


def trace_llm_call(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    response: dict[str, Any],
    duration_ms: float,
) -> None:
    """Record an LLM call in Langfuse."""
    if not _langfuse:
        return

    try:
        trace = _langfuse.trace(
            name=f"chat/{provider}",
            metadata={"provider": provider, "model": model},
        )
        trace.generation(
            name=f"{provider}/{model}",
            model=model,
            input=messages,
            output=response.get("content", ""),
            usage={
                "input": response.get("usage", {}).get("prompt_tokens", 0),
                "output": response.get("usage", {}).get("completion_tokens", 0),
            },
            metadata={
                "provider": provider,
                "duration_ms": duration_ms,
            },
        )
    except Exception as e:
        logger.warning(f"Langfuse trace failed: {e}")


def flush_langfuse() -> None:
    """Flush pending Langfuse events."""
    if _langfuse:
        try:
            _langfuse.flush()
        except Exception:
            pass
