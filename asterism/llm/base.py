"""Base LLM provider interface for the agent framework."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage

from asterism.core.prompt_loader import SystemPromptLoader


@dataclass
class LLMResponse:
    """Response from LLM including content and usage metadata."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StructuredLLMResponse(LLMResponse):
    """Response from LLM structured output including parsed model and usage."""

    parsed: Any = None
    """The parsed Pydantic model instance."""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    This class provides a message-based interface for LLM interactions,
    with built-in support for loading system prompts from SOUL.md and
    AGENT.md files. System prompts are loaded fresh on each call to
    ensure runtime updates are reflected.

    Attributes:
        prompt_loader: Optional SystemPromptLoader for loading SOUL.md and AGENT.md.
                      If None, no additional system prompts are prepended.
    """

    def __init__(
        self,
        prompt_loader: SystemPromptLoader | None = None,
    ):
        """
        Initialize the LLM provider.

        Args:
            prompt_loader: Optional SystemPromptLoader instance for loading
                          SOUL.md and AGENT.md. If provided, the content from
                          these files will be prepended to all LLM calls as
                          a SystemMessage. If None, no additional system
                          prompts are added.
        """
        self.prompt_loader = prompt_loader

    @abstractmethod
    def invoke(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> str:
        """
        Invoke the LLM with a text prompt or message list.

        If a string is provided, it will be converted to a HumanMessage.
        If a prompt_loader is configured, SOUL.md and AGENT.md content
        will be prepended as a SystemMessage.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
                    When a string is provided, it will be wrapped as a HumanMessage.
            **kwargs: Additional provider-specific parameters.
                      Common parameters include:
                      - system_message: Additional system prompt to prepend
                      - temperature: Generation temperature
                      - max_tokens: Maximum tokens to generate

        Returns:
            The LLM's text response.
        """
        pass

    @abstractmethod
    def invoke_with_usage(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Invoke the LLM and return response with token usage.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse containing content and usage metadata.
        """
        pass

    @abstractmethod
    def invoke_structured(
        self,
        prompt: str | list[BaseMessage],
        schema: type,
        **kwargs,
    ) -> StructuredLLMResponse:
        """
        Invoke the LLM with a structured output request.

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
        pass

    async def astream(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> AsyncGenerator[str]:
        """
        Stream LLM response tokens asynchronously.

        This is a base implementation that falls back to invoke().
        Subclasses should override this with native streaming support.

        Args:
            prompt: Either a text prompt (str) or a list of messages.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Tokens (strings) as they are generated.
        """
        # Default implementation: invoke and yield full response as single chunk
        result = self.invoke(prompt, **kwargs)
        yield result

    def _build_messages(
        self,
        prompt: str | list[BaseMessage],
        **kwargs,
    ) -> list[BaseMessage]:
        """
        Build the full message list with system prompts prepended.

        This method:
        1. Loads SOUL.md and AGENT.md if prompt_loader is configured
        2. Wraps string prompts as HumanMessage
        3. Prepends the loaded system prompt
        4. Appends any additional system message from kwargs

        Args:
            prompt: Either a text prompt or list of messages.
            **kwargs: May contain 'system_message' for additional system content.

        Returns:
            List of messages with system prompts prepended.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Load system prompts if loader is configured
        system_messages: list[BaseMessage] = []
        if self.prompt_loader is not None:
            system_content = self.prompt_loader.load()
            system_messages.append(SystemMessage(content=system_content))

        # Add any additional system message from kwargs
        additional_system = kwargs.pop("system_message", None)
        if additional_system:
            if isinstance(additional_system, str):
                system_messages.append(SystemMessage(content=additional_system))
            elif isinstance(additional_system, SystemMessage):
                system_messages.append(additional_system)
            elif isinstance(additional_system, list):
                for msg in additional_system:
                    if isinstance(msg, str):
                        system_messages.append(SystemMessage(content=msg))
                    elif isinstance(msg, BaseMessage):
                        system_messages.append(msg)

        # Convert string prompt to HumanMessage if needed
        if isinstance(prompt, str):
            user_messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        elif isinstance(prompt, list):
            user_messages = prompt
        else:
            user_messages = [prompt]

        # Combine: system messages first, then user messages
        return system_messages + user_messages

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name/version being used."""
        pass

    def set_model(self, model: str) -> None:
        """Set the model for this provider.

        This allows dynamic model switching per-request.

        Args:
            model: Model name to use for subsequent calls.
        """
        # Base implementation - subclasses should override if they support dynamic model switching
        pass
