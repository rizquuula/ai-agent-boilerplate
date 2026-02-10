"""LLM Provider Router with primary-first fallback."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import BaseMessage

from asterism.config import Config

from .base import BaseLLMProvider, LLMResponse, StructuredLLMResponse
from .exceptions import AllProvidersFailedError
from .factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class LLMProviderRouter(BaseLLMProvider):
    """Routes LLM calls across multiple providers with primary-first fallback.

    This router implements a sequential fallback strategy where:
    1. The primary provider is tried first
    2. On failure, each fallback provider is tried in order
    3. If all providers fail, an AllProvidersFailedError is raised

    Model format: "provider_name/model_path" or just "model_path" (uses default provider)

    Attributes:
        config: Configuration object with provider and fallback settings
        providers: Dictionary of provider name -> provider instance
    """

    def __init__(self, config: Config | None = None):
        """Initialize the provider router.

        Args:
            config: Configuration object. If None, creates a new Config instance.
        """
        super().__init__(prompt_loader=None)
        self.config = config or Config()
        self.providers: dict[str, BaseLLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Create provider instances from configuration."""
        for provider_config in self.config.data.models.provider:
            try:
                provider = LLMProviderFactory.create_provider(provider_config)
                self.providers[provider_config.name] = provider
                logger.debug(f"Initialized provider: {provider_config.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {provider_config.name}: {e}")

    def invoke(self, prompt: str | list[BaseMessage], **kwargs: Any) -> str:
        """Invoke LLM with primary-first fallback.

        Args:
            prompt: Text or messages to send to the LLM
            **kwargs: Additional parameters including:
                - model: Model identifier (provider/model or just model)
                - Other provider-specific parameters

        Returns:
            LLM response string

        Raises:
            AllProvidersFailedError: If all providers in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        provider_chain = self._build_provider_chain(model)
        provider_names = [p.name for p in provider_chain]

        if not provider_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=provider_names,
            )

        last_error: Exception | None = None

        for provider in provider_chain:
            try:
                # Extract model name for this provider (remove provider prefix)
                provider_model = self._extract_model_for_provider(model, provider.name)
                if provider_model:
                    kwargs["model"] = provider_model
                else:
                    kwargs.pop("model", None)

                result = provider.invoke(prompt, **kwargs)
                logger.debug(f"Provider succeeded: {provider.name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All providers failed after trying {len(provider_chain)} provider(s).",
            last_error=last_error,
            provider_chain=provider_names,
        )

    def invoke_with_usage(
        self,
        prompt: str | list[BaseMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke LLM with usage tracking and primary-first fallback.

        Args:
            prompt: Text or messages to send to the LLM
            **kwargs: Additional parameters including:
                - model: Model identifier (provider/model or just model)

        Returns:
            LLMResponse containing content and usage metadata

        Raises:
            AllProvidersFailedError: If all providers in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        provider_chain = self._build_provider_chain(model)
        provider_names = [p.name for p in provider_chain]

        if not provider_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=provider_names,
            )

        last_error: Exception | None = None

        for provider in provider_chain:
            try:
                # Extract model name for this provider (remove provider prefix)
                provider_model = self._extract_model_for_provider(model, provider.name)
                if provider_model:
                    kwargs["model"] = provider_model
                else:
                    kwargs.pop("model", None)

                result = provider.invoke_with_usage(prompt, **kwargs)
                logger.debug(f"Provider succeeded: {provider.name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All providers failed after trying {len(provider_chain)} provider(s).",
            last_error=last_error,
            provider_chain=provider_names,
        )

    def invoke_structured(
        self,
        prompt: str | list[BaseMessage],
        schema: type,
        **kwargs: Any,
    ) -> StructuredLLMResponse:
        """Invoke LLM with structured output and primary-first fallback.

        Args:
            prompt: Text or messages to send to the LLM
            schema: Pydantic model for structured output
            **kwargs: Additional parameters including:
                - model: Model identifier (provider/model or just model)

        Returns:
            StructuredLLMResponse containing parsed model and usage metadata

        Raises:
            AllProvidersFailedError: If all providers in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        provider_chain = self._build_provider_chain(model)
        provider_names = [p.name for p in provider_chain]

        if not provider_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=provider_names,
            )

        last_error: Exception | None = None

        for provider in provider_chain:
            try:
                # Extract model name for this provider (remove provider prefix)
                provider_model = self._extract_model_for_provider(model, provider.name)
                if provider_model:
                    kwargs["model"] = provider_model
                else:
                    kwargs.pop("model", None)

                result = provider.invoke_structured(prompt, schema, **kwargs)
                logger.debug(f"Provider succeeded: {provider.name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All providers failed after trying {len(provider_chain)} provider(s).",
            last_error=last_error,
            provider_chain=provider_names,
        )

    def _build_provider_chain(self, model: str | None) -> list[BaseLLMProvider]:
        """Build the provider chain for a model request.

        The chain is built as: primary → fallback[0] → fallback[1] → ...

        Args:
            model: Model identifier (provider/model format) or None to use default

        Returns:
            List of providers in order of priority
        """
        chain: list[BaseLLMProvider] = []
        seen: set[str] = set()

        # Parse model identifier to get primary provider name
        if model and "/" in model:
            primary_provider_name = model.split("/", 1)[0]
        else:
            # Use default from config
            default_model = self.config.data.models.default
            if "/" in default_model:
                primary_provider_name = default_model.split("/", 1)[0]
            else:
                primary_provider_name = default_model

        # Add primary provider
        if primary_provider_name and primary_provider_name not in seen:
            primary = self.providers.get(primary_provider_name)
            if primary:
                chain.append(primary)
                seen.add(primary_provider_name)

        # Add fallback providers from config
        for fallback_model in self.config.data.models.fallback:
            if "/" in fallback_model:
                fallback_provider_name, _ = fallback_model.split("/", 1)
            else:
                fallback_provider_name = fallback_model

            if fallback_provider_name and fallback_provider_name not in seen:
                provider = self.providers.get(fallback_provider_name)
                if provider:
                    chain.append(provider)
                    seen.add(fallback_provider_name)

        return chain

    def _extract_model_for_provider(self, model: str | None, provider_name: str) -> str | None:
        """Extract the model name for a specific provider.

        Args:
            model: Full model string (provider/model) or just model name
            provider_name: Name of the provider to extract model for

        Returns:
            Model name without provider prefix, or None if not applicable
        """
        if not model:
            return None

        if "/" in model:
            parts = model.split("/", 1)
            if parts[0] == provider_name:
                return parts[1]
            # Model is for a different provider
            return None

        # No provider prefix, return as-is
        return model

    async def astream(
        self,
        prompt: str | list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Stream LLM response with primary-first fallback.

        Args:
            prompt: Text or messages to send to the LLM
            **kwargs: Additional parameters including:
                - model: Model identifier (provider/model or just model)
                - Other provider-specific parameters

        Yields:
            Tokens (strings) as they are generated.

        Raises:
            AllProvidersFailedError: If all providers in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        provider_chain = self._build_provider_chain(model)
        provider_names = [p.name for p in provider_chain]

        if not provider_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=provider_names,
            )

        last_error: Exception | None = None

        for provider in provider_chain:
            try:
                # Extract model name for this provider (remove provider prefix)
                provider_model = self._extract_model_for_provider(model, provider.name)
                if provider_model:
                    kwargs["model"] = provider_model
                else:
                    kwargs.pop("model", None)

                logger.debug(f"Streaming with provider: {provider.name}")
                async for token in provider.astream(prompt, **kwargs):
                    yield token
                return  # Successfully streamed, exit

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed during streaming: {e}")
                continue

        raise AllProvidersFailedError(
            f"All providers failed during streaming after trying {len(provider_chain)} provider(s).",
            last_error=last_error,
            provider_chain=provider_names,
        )

    def set_model(self, model: str) -> None:
        """Set the model is not applicable for router (model is per-request).

        Args:
            model: Ignored for router.
        """
        # Router doesn't have a fixed model - it's determined per-request
        pass

    @property
    def name(self) -> str:
        """Name of the LLM provider router."""
        return "Router"

    @property
    def model(self) -> str:
        """Model name being used (returns default from config)."""
        return self.config.data.models.default
