"""Shared fixtures for unit tests."""

import os

import pytest
from pydantic import BaseModel

from agent.llm.base import BaseLLMProvider
from agent.llm.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider() -> BaseLLMProvider:
    """Create an OpenAI provider instance with custom base URL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        pytest.skip("OPENAI_MODEL not set in environment")

    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        pytest.skip("OPENAI_BASE_URL not set in environment")

    provider = OpenAIProvider(model=model, base_url=base_url, api_key=api_key)
    return provider


@pytest.fixture
def simple_schema():
    """A simple Pydantic schema for structured output testing."""

    class SimpleResponse(BaseModel):
        answer: str
        confidence: float

    return SimpleResponse


@pytest.fixture
def complex_schema():
    """A more complex Pydantic schema for structured output testing."""

    class ComplexResponse(BaseModel):
        question: str
        answer: str
        reasoning: str
        confidence: float
        tags: list[str]

    return ComplexResponse


@pytest.fixture
def test_prompt():
    """A simple test prompt."""
    return "What is the capital of France?"


@pytest.fixture
def test_structured_prompt():
    """A prompt designed for structured output."""
    return "Answer this question with a JSON response: What is 2+2? Provide your answer and confidence level."
