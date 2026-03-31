"""Tests for Langfuse tracing."""
import pytest
from unittest.mock import patch, MagicMock
import life_core.langfuse_tracing as lf_mod
from life_core.langfuse_tracing import init_langfuse, trace_llm_call, flush_langfuse


def setup_function():
    """Reset global langfuse client before each test."""
    lf_mod._langfuse = None


def test_init_no_env():
    with patch.dict("os.environ", {}, clear=True):
        init_langfuse()
    assert lf_mod._langfuse is None


def test_init_partial_env_missing_host():
    with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}, clear=False):
        lf_mod._langfuse = None
        init_langfuse()
    assert lf_mod._langfuse is None


def test_init_with_all_env_but_import_error():
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "LANGFUSE_HOST": "http://langfuse:3000",
    }
    with patch.dict("os.environ", env):
        with patch.dict("sys.modules", {"langfuse": None}):
            lf_mod._langfuse = None
            init_langfuse()
    # import error => client remains None
    assert lf_mod._langfuse is None


def test_trace_no_client():
    lf_mod._langfuse = None
    # Should not crash when langfuse not initialized
    trace_llm_call(
        "ollama", "qwen3:4b",
        [{"role": "user", "content": "hi"}],
        {"content": "hello"},
        234.0,
    )


def test_flush_no_client():
    lf_mod._langfuse = None
    flush_langfuse()  # Must not raise


def test_flush_calls_flush_on_client():
    mock_client = MagicMock()
    lf_mod._langfuse = mock_client
    flush_langfuse()
    mock_client.flush.assert_called_once()


def test_trace_calls_trace_on_client():
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace
    lf_mod._langfuse = mock_client

    trace_llm_call(
        "claude", "claude-3-5-haiku",
        [{"role": "user", "content": "hello"}],
        {"content": "world", "usage": {"prompt_tokens": 5, "completion_tokens": 10}},
        150.0,
    )

    mock_client.trace.assert_called_once()
    mock_trace.generation.assert_called_once()


def test_trace_handles_exception():
    mock_client = MagicMock()
    mock_client.trace.side_effect = Exception("network error")
    lf_mod._langfuse = mock_client

    # Must not propagate
    trace_llm_call("ollama", "llama3", [], {}, 0.0)


def test_flush_handles_exception():
    mock_client = MagicMock()
    mock_client.flush.side_effect = Exception("flush error")
    lf_mod._langfuse = mock_client

    flush_langfuse()  # Must not raise
