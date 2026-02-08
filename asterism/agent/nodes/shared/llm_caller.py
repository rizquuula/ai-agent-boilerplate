"""Centralized LLM invocation with standardized logging and timing."""

import time
from dataclasses import dataclass
from typing import Any, TypeVar

from asterism.agent.models import LLMUsage
from asterism.agent.utils import log_llm_call, log_llm_call_start
from asterism.llm.base import BaseLLMProvider

T = TypeVar("T")


@dataclass
class LLMCallResult:
    """Result of an LLM call with parsed content and usage info."""

    parsed: Any
    usage: LLMUsage
    duration_ms: float


class LLMCallError(Exception):
    """Error raised when LLM call fails."""

    pass


class LLMCaller:
    """Wrapper for LLM calls with consistent logging and error handling.

    This class centralizes all LLM invocation logic including:
    - Timing measurement
    - Structured logging
    - Usage tracking
    - Error handling
    """

    def __init__(self, llm: BaseLLMProvider, node_name: str):
        self.llm = llm
        self.node_name = node_name
        self._logger = __import__("logging").getLogger(__name__)

    def call_structured(self, messages: list, schema: type[T], action: str) -> LLMCallResult:
        """Make a structured LLM call with full logging.

        Args:
            messages: List of messages to send to LLM
            schema: Pydantic model class for structured output
            action: Description of the action for logging

        Returns:
            LLMCallResult with parsed data, usage info, and timing

        Raises:
            LLMCallError: If the LLM call fails
        """
        prompt_preview = self._extract_preview(messages)

        log_llm_call_start(
            logger=self._logger,
            node_name=self.node_name,
            model=self.llm.model,
            action=action,
            prompt_preview=prompt_preview,
        )

        start_time = time.perf_counter()

        try:
            response = self.llm.invoke_structured(messages, schema)
            duration_ms = (time.perf_counter() - start_time) * 1000

            usage = LLMUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                model=self.llm.model,
                node_name=self.node_name,
            )

            log_llm_call(
                logger=self._logger,
                node_name=self.node_name,
                model=self.llm.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                duration_ms=duration_ms,
                prompt_preview=prompt_preview,
                response_preview=str(response.parsed.model_dump())[:500] if response.parsed else None,
                success=True,
            )

            return LLMCallResult(
                parsed=response.parsed,
                usage=usage,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Try to extract token counts from partial response if available
            prompt_tokens = getattr(getattr(e, "response", None), "prompt_tokens", 0)
            completion_tokens = getattr(getattr(e, "response", None), "completion_tokens", 0)

            log_llm_call(
                logger=self._logger,
                node_name=self.node_name,
                model=self.llm.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                prompt_preview=prompt_preview,
                success=False,
                error=str(e),
            )

            raise LLMCallError(f"LLM call failed for {action}: {e}") from e

    def call_text(self, messages: list, action: str) -> LLMCallResult:
        """Make a text-based LLM call with full logging.

        Args:
            messages: Message or list of messages (can be string for simple prompts)
            action: Description of the action for logging

        Returns:
            LLMCallResult with text content, usage info, and timing

        Raises:
            LLMCallError: If the LLM call fails
        """
        # Handle both single string and list of messages
        if isinstance(messages, str):
            prompt_preview = messages[:200]
        else:
            prompt_preview = self._extract_preview(messages)

        log_llm_call_start(
            logger=self._logger,
            node_name=self.node_name,
            model=self.llm.model,
            action=action,
            prompt_preview=prompt_preview,
        )

        start_time = time.perf_counter()

        try:
            response = self.llm.invoke_with_usage(messages)
            duration_ms = (time.perf_counter() - start_time) * 1000

            usage = LLMUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                model=self.llm.model,
                node_name=self.node_name,
            )

            log_llm_call(
                logger=self._logger,
                node_name=self.node_name,
                model=self.llm.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                duration_ms=duration_ms,
                prompt_preview=prompt_preview,
                response_preview=response.content[:500] if response.content else None,
                success=True,
            )

            return LLMCallResult(
                parsed=response.content,
                usage=usage,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            prompt_tokens = getattr(getattr(e, "response", None), "prompt_tokens", 0)
            completion_tokens = getattr(getattr(e, "response", None), "completion_tokens", 0)

            log_llm_call(
                logger=self._logger,
                node_name=self.node_name,
                model=self.llm.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                prompt_preview=prompt_preview,
                success=False,
                error=str(e),
            )

            raise LLMCallError(f"LLM call failed for {action}: {e}") from e

    def _extract_preview(self, messages: list) -> str:
        """Extract a preview string from messages for logging."""
        if not messages:
            return ""

        previews = []
        for msg in messages[:2]:  # Preview first 2 messages
            content = getattr(msg, "content", str(msg))
            previews.append(content[:150])

        return " | ".join(previews)
