"""Unit tests for OpenAI provider using real API calls."""

import pytest

from agent.llm.base import BaseLLMProvider
from src.agent.llm.openai_provider import OpenAIProvider


def test_invoke_simple_prompt(openai_provider: BaseLLMProvider, test_prompt: str):
    """Test basic prompt invocation."""
    response = openai_provider.invoke(test_prompt)

    assert isinstance(response, str)
    assert len(response.strip()) > 0
    # Should contain something related to Paris (capital of France)
    assert "paris" in response.lower() or "france" in response.lower()


def test_invoke_with_parameters(openai_provider: BaseLLMProvider):
    """Test invoke with additional parameters."""
    prompt = "Say 'Hello World' and nothing else."
    response = openai_provider.invoke(
        prompt,
        temperature=0.1,  # Low temperature for consistent output
        max_tokens=50
    )

    assert isinstance(response, str)
    assert len(response.strip()) > 0
    # Should be very close to just "Hello World"
    assert "hello world" in response.lower()


def test_invoke_error_handling():
    """Test error handling in invoke method."""
    # Create provider with invalid API key
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="invalid-key")

    with pytest.raises(RuntimeError, match="OpenAI API error"):
        provider.invoke("Test prompt")


def test_invoke_structured_basic(
    openai_provider: BaseLLMProvider,
    test_structured_prompt: str,
    simple_schema: object,
):
    """Test basic structured output invocation."""
    response = openai_provider.invoke_structured(test_structured_prompt, simple_schema)

    assert isinstance(response, simple_schema)
    assert hasattr(response, 'answer')
    assert hasattr(response, 'confidence')
    assert isinstance(response.answer, str)
    assert isinstance(response.confidence, float)
    # Should contain answer to 2+2
    assert "4" in response.answer or "four" in response.answer.lower()


def test_invoke_structured_complex_schema(
    openai_provider: BaseLLMProvider,
    complex_schema: object,
):
    """Test structured output with complex schema."""
    prompt = """
    Answer this question in detail: What is the meaning of life?
    Provide your response with question, answer, reasoning, confidence score, and relevant tags.
    """

    response = openai_provider.invoke_structured(prompt, complex_schema)

    assert isinstance(response, complex_schema)
    assert hasattr(response, 'question')
    assert hasattr(response, 'answer')
    assert hasattr(response, 'reasoning')
    assert hasattr(response, 'confidence')
    assert hasattr(response, 'tags')

    assert isinstance(response.question, str)
    assert isinstance(response.answer, str)
    assert isinstance(response.reasoning, str)
    assert isinstance(response.confidence, float)
    assert isinstance(response.tags, list)

    # Basic validation
    assert len(response.question.strip()) > 0
    assert len(response.answer.strip()) > 0
    assert len(response.reasoning.strip()) > 0
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.tags) > 0


def test_invoke_structured_error_handling():
    """Test error handling in structured invoke method."""
    # Create provider with invalid API key
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="invalid-key")

    from pydantic import BaseModel
    class TestSchema(BaseModel):
        result: str

    with pytest.raises(RuntimeError, match="OpenAI structured output error"):
        provider.invoke_structured("Test prompt", TestSchema)


def test_invoke_structured_with_kwargs(openai_provider: BaseLLMProvider, simple_schema):
    """Test structured invoke with additional parameters."""
    prompt = "What is 3+5? Answer with just the number."
    response = openai_provider.invoke_structured(
        prompt,
        simple_schema,
        temperature=0.0,  # Deterministic output
        max_tokens=100
    )

    assert isinstance(response, simple_schema)
    assert "8" in response.answer or "eight" in response.answer.lower()

