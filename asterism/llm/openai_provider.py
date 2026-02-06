"""OpenAI LLM provider implementation."""

import os

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

    def invoke_structured(
        self,
        prompt: str | list[BaseMessage],
        schema: type,
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
            **kwargs: Additional provider-specific parameters.

        Returns:
            StructuredLLMResponse containing parsed model and usage metadata.
        """
        try:
            # Build full message list with system prompts (SOUL + AGENT)
            messages = self._build_messages(prompt, **kwargs)

            # Create output parser for the schema
            parser = PydanticOutputParser(pydantic_object=schema)

            # Create a custom chain that captures raw response and usage
            # We use a simple approach: invoke client directly, then parse
            raw_response = self.client.invoke(messages, **kwargs)

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
            parsed_result = parser.parse(content)

            return StructuredLLMResponse(
                content=content,
                parsed=parsed_result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI structured output error: {str(e)}")

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
        return "OpenAI"

    @property
    def model(self) -> str:
        """Model name/version being used."""
        return self._model
