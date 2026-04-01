"""Tests for vLLM routing in LiteLLMProvider."""
import pytest
from unittest.mock import AsyncMock, patch

from life_core.router.providers.litellm_provider import LiteLLMProvider


def test_build_call_kwargs_routes_vllm_model():
    """vLLM models get api_base injected."""
    provider = LiteLLMProvider(
        models=["openai/qwen-27b-awq", "ollama/llama3"],
        ollama_api_base="http://cils:11434",
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq", "openai/mascarade-stm32"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/qwen-27b-awq", {})
    assert kwargs["api_base"] == "http://kxkm:8000"


def test_build_call_kwargs_routes_ollama_model():
    """Ollama models still get ollama api_base."""
    provider = LiteLLMProvider(
        models=["openai/qwen-27b-awq", "ollama/llama3"],
        ollama_api_base="http://cils:11434",
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("ollama/llama3", {})
    assert kwargs["api_base"] == "http://cils:11434"


def test_build_call_kwargs_cloud_model_no_api_base():
    """Cloud models get no api_base override."""
    provider = LiteLLMProvider(
        models=["anthropic/claude-sonnet-4-20250514"],
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("anthropic/claude-sonnet-4-20250514", {})
    assert "api_base" not in kwargs


def test_build_call_kwargs_vllm_lora_model():
    """LoRA model names route to vLLM."""
    provider = LiteLLMProvider(
        models=["openai/mascarade-stm32"],
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/mascarade-stm32"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/mascarade-stm32", {})
    assert kwargs["api_base"] == "http://kxkm:8000"


def test_vllm_defaults_to_empty():
    """No vLLM config = no vLLM routing."""
    provider = LiteLLMProvider(models=["openai/gpt-4o"])
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/gpt-4o", {})
    assert "api_base" not in kwargs
