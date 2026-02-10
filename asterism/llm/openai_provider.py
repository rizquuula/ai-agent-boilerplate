"""OpenAI LLM provider implementation."""

import os
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from asterism.core.prompt_loader import SystemPromptLoader

from .base import BaseLLMProvider, LLMResponse, StructuredLLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain.

    This provider supports both simple string prompts and full message-based
    conversations. When a SystemPromptLoader is configured, SOUL.md and
    AGENT.md content will be automatically prepended to all LLM calls.
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt_loader: SystemPromptLoader | None = None,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name/version
            base_url: OpenAI base URL (if None, uses default OpenAI URL)
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            prompt_loader: Optional SystemPromptLoader for loading SOUL.md and AGENT.md.
                          If provided, these files' content will be prepended to all LLM calls.
            **kwargs: Additional LangChain ChatOpenAI parameters
        """
        super().__init__(prompt_loader=prompt_loader)
        self._name = provider_name
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        # Initialize LangChain OpenAI client
        self.client = ChatOpenAI(model=model, base_url=self._base_url, api_key=self._api_key, **kwargs)

    def invoke(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> str:
        """
        Invoke OpenAI LLM with a text prompt or message list.

        If a string is provided, it will be converted to a HumanMessage.
        If a prompt_loader is configured, SOUL.md and AGENT.md content
        will be prepended as a SystemMessage.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The LLM's text response.
        """
        # Build full message list with system prompts
        messages = self._build_messages(prompt, **kwargs)

        try:
            response = self.client.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def invoke_with_usage(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Invoke OpenAI LLM and return response with token usage.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse containing content and usage metadata.
        """
        # Build full message list with system prompts
        messages = self._build_messages(prompt, **kwargs)

        try:
            response = self.client.invoke(messages, **kwargs)

            # Extract usage information if available
            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens = usage.get("input_tokens", 0)
                completion_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            else:
                # Fallback: estimate tokens (rough approximation)
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            return LLMResponse(
                content=response.content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def _extract_json_from_text(self, text: str) -> str | None:
        """
        Extract JSON from text that may contain markdown code blocks or other content.

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Extracted JSON string or None if extraction fails.
        """
        # Try to find JSON in markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        if matches:
            return matches[0].strip()

        # Try to find JSON between curly braces
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
        except Exception:
            pass

        return None

    def invoke_structured(
        self,
        prompt: str | list[BaseMessage],
        schema: type,
        max_retries: int = 3,
        **kwargs,
    ) -> StructuredLLMResponse:
        """
        Invoke OpenAI LLM with structured output.

        If a string is provided, it will be converted to a HumanMessage.
        If a prompt_loader is configured, SOUL.md and AGENT.md content
        will be prepended as a SystemMessage.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            schema: Pydantic model or type for structured output.
            max_retries: Maximum number of retry attempts for parsing failures.
            **kwargs: Additional provider-specific parameters.

        Returns:
            StructuredLLMResponse containing parsed model and usage metadata.
        """
        # Build full message list with system prompts (SOUL + AGENT)
        messages = self._build_messages(prompt, **kwargs)

        # Create output parser for the schema
        parser = PydanticOutputParser(pydantic_object=schema)

        last_error = None
        for attempt in range(max_retries):
            try:
                # Use JSON mode for structured output (OpenAI-specific)
                response_format = {"type": "json_object"}

                # Invoke with JSON mode
                raw_response = self.client.invoke(
                    messages,
                    # response_format=response_format,
                    **kwargs,
                )

                # Extract usage information from raw response
                usage = getattr(raw_response, "usage_metadata", None)
                if usage:
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                else:
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0

                # Parse the content using the parser
                content = raw_response.content

                try:
                    parsed_result = parser.parse(content)
                except Exception as parse_error:
                    # Try to extract JSON from markdown or other formatting
                    extracted_json = self._extract_json_from_text(content)
                    if extracted_json:
                        try:
                            # Parse the extracted JSON
                            parsed_result = parser.parse(extracted_json)
                            content = extracted_json  # Use the cleaned content
                        except Exception:
                            # If extraction still fails, raise original error
                            raise parse_error
                    else:
                        raise parse_error

                return StructuredLLMResponse(
                    content=content,
                    parsed=parsed_result,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, include raw content in error if available
                    error_msg = f"OpenAI structured output error after {max_retries} attempts: {str(e)}"
                    if "content" in locals():
                        error_msg += f"\n\nRaw LLM output:\n{content[:2000]}"
                    raise RuntimeError(error_msg)

    async def astream(
        self,
        prompt: str | list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """
        Stream LLM response tokens asynchronously using LangChain's astream.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            **kwargs: Additional provider-specific parameters.
                - model: Override the model for this request

        Yields:
            Tokens (strings) as they are generated.
        """
        # Build full message list with system prompts
        messages = self._build_messages(prompt, **kwargs)

        # Allow model override per-request
        model = kwargs.get("model", self._model)

        # Create client with potentially different model
        if model != self._model:
            client = ChatOpenAI(
                model=model,
                base_url=self._base_url,
                api_key=self._api_key,
                streaming=True,
            )
        else:
            # Use existing client but enable streaming
            client = self.client

        try:
            async for chunk in client.astream(messages, **kwargs):
                content = chunk.content
                if content:
                    yield content
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {str(e)}") from e

    def set_model(self, model: str) -> None:
        """Set the model for this provider.

        Args:
            model: Model name to use for subsequent calls.
        """
        self._model = model
        # Re-initialize client with new model
        self.client = ChatOpenAI(
            model=model,
            base_url=self._base_url,
            api_key=self._api_key,
        )

    def _messages_to_text(self, messages: list[BaseMessage]) -> str:
        """
        Convert a list of messages to a text representation.

        This is used for structured output where we need a single
        text prompt for the template.

        Args:
            messages: List of messages to convert.

        Returns:
            A text representation of the messages.
        """
        text_parts = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "").lower()
            content = getattr(msg, "content", str(msg))
            text_parts.append(f"[{role}]: {content}")
        return "\n".join(text_parts)

    @property
    def name(self) -> str:
        """Name of the LLM provider."""
        return self._name

    @property
    def model(self) -> str:
        """Model name/version being used."""
        return self._model
