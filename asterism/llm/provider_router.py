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
    1. The primary model (from request or config default) is tried first
    2. On failure, each fallback model is tried in order
    3. If all models fail, an AllProvidersFailedError is raised

    Model format: "provider_name/model_path" or just "model_path" (uses default provider)

    The fallback chain is built from models, not providers, allowing multiple
    models from the same provider to be used as fallbacks.

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
            AllProvidersFailedError: If all models in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        model_chain = self._build_model_chain(model)
        model_names = [f"{p.name}/{m}" for p, m in model_chain]

        if not model_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=model_names,
            )

        last_error: Exception | None = None

        for provider, model_name in model_chain:
            try:
                kwargs["model"] = model_name
                result = provider.invoke(prompt, **kwargs)
                logger.debug(f"Model succeeded: {provider.name}/{model_name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Model {provider.name}/{model_name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All models failed after trying {len(model_chain)} model(s).",
            last_error=last_error,
            provider_chain=model_names,
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
            AllProvidersFailedError: If all models in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        model_chain = self._build_model_chain(model)
        model_names = [f"{p.name}/{m}" for p, m in model_chain]

        if not model_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=model_names,
            )

        last_error: Exception | None = None

        for provider, model_name in model_chain:
            try:
                kwargs["model"] = model_name
                result = provider.invoke_with_usage(prompt, **kwargs)
                logger.debug(f"Model succeeded: {provider.name}/{model_name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Model {provider.name}/{model_name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All models failed after trying {len(model_chain)} model(s).",
            last_error=last_error,
            provider_chain=model_names,
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
            AllProvidersFailedError: If all models in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        model_chain = self._build_model_chain(model)
        model_names = [f"{p.name}/{m}" for p, m in model_chain]

        if not model_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=model_names,
            )

        last_error: Exception | None = None

        for provider, model_name in model_chain:
            try:
                kwargs["model"] = model_name
                result = provider.invoke_structured(prompt, schema, **kwargs)
                logger.debug(f"Model succeeded: {provider.name}/{model_name}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Model {provider.name}/{model_name} failed: {e}")
                continue

        raise AllProvidersFailedError(
            f"All models failed after trying {len(model_chain)} model(s).",
            last_error=last_error,
            provider_chain=model_names,
        )

    def _build_model_chain(self, primary_model: str | None) -> list[tuple[BaseLLMProvider, str]]:
        """Build the model chain for a request.

        The chain is built as: primary → fallback[0] → fallback[1] → ...
        Each entry is a tuple of (provider_instance, model_name).

        Unlike the previous provider-based chain, this allows multiple models
        from the same provider to be used as fallbacks.

        Args:
            primary_model: Primary model identifier (provider/model format) or None

        Returns:
            List of (provider, model_name) tuples in order of priority
        """
        chain: list[tuple[BaseLLMProvider, str]] = []

        # Build list of model strings to try: primary + fallbacks
        model_strings: list[str] = []

        # Add primary model (from request or config default)
        if primary_model:
            model_strings.append(primary_model)

        # Add fallback models from config
        for fallback_model in self.config.data.models.fallback:
            if fallback_model not in model_strings:
                model_strings.append(fallback_model)

        # Build chain of (provider, model_name) tuples
        for model_string in model_strings:
            provider_name, model_name = self._parse_model_string(model_string)

            provider = self.providers.get(provider_name)
            if provider:
                chain.append((provider, model_name))
            else:
                logger.warning(f"Provider '{provider_name}' not found for model '{model_string}'")

        return chain

    def _parse_model_string(self, model_string: str) -> tuple[str, str]:
        """Parse a model string into provider name and model name.

        Args:
            model_string: Model identifier in format "provider/model" or just "model"

        Returns:
            Tuple of (provider_name, model_name)
        """
        if "/" in model_string:
            parts = model_string.split("/", 1)
            return parts[0], parts[1]

        # No provider prefix, use default provider from config
        default_model = self.config.data.models.default
        if "/" in default_model:
            default_provider = default_model.split("/", 1)[0]
            return default_provider, model_string

        # Default model also has no provider, use model string as-is
        # This will likely fail but preserves backward compatibility
        return model_string, model_string

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
            AllProvidersFailedError: If all models in the chain fail
        """
        model = kwargs.get("model", self.config.data.models.default)
        model_chain = self._build_model_chain(model)
        model_names = [f"{p.name}/{m}" for p, m in model_chain]

        if not model_chain:
            raise AllProvidersFailedError(
                "No providers available in the chain",
                provider_chain=model_names,
            )

        last_error: Exception | None = None

        for provider, model_name in model_chain:
            try:
                kwargs["model"] = model_name
                logger.debug(f"Streaming with model: {provider.name}/{model_name}")
                async for token in provider.astream(prompt, **kwargs):
                    yield token
                return  # Successfully streamed, exit

            except Exception as e:
                last_error = e
                logger.warning(f"Model {provider.name}/{model_name} failed during streaming: {e}")
                continue

        raise AllProvidersFailedError(
            f"All models failed during streaming after trying {len(model_chain)} model(s).",
            last_error=last_error,
            provider_chain=model_names,
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
