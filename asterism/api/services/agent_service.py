"""Agent lifecycle management service."""

import logging
from typing import Any

from asterism.agent import Agent
from asterism.config import Config
from asterism.llm import LLMProviderRouter
from asterism.mcp.executor import MCPExecutor

from ..models import ChatCompletionRequest

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing Agent lifecycle per request.

    Design: Stateless - creates fresh Agent for each request.
    No session persistence, no SQLite checkpointing for API mode.
    """

    def __init__(
        self,
        llm_router: LLMProviderRouter,
        mcp_executor: MCPExecutor,
        config: Config,
    ):
        """Initialize the agent service.

        Args:
            llm_router: LLM provider router for fallback support
            mcp_executor: MCP executor for tool calls
            config: Configuration instance
        """
        self.llm_router = llm_router
        self.mcp_executor = mcp_executor
        self.config = config

    async def run_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> dict[str, Any]:
        """Run a single completion (non-streaming).

        Args:
            request: The chat completion request
            request_id: Unique request identifier

        Returns:
            Dictionary containing the agent response and metadata
        """
        # Create fresh agent for this request (no checkpointing for API mode)
        agent = Agent(
            llm=self.llm_router,
            mcp_executor=self.mcp_executor,
            db_path=None,  # Disable checkpointing for API requests
            workspace_root=self.config.workspace_path,
        )

        try:
            # Extract the last user message as the input
            user_message = self._extract_last_user_message(request.messages)

            # Set model if specified in request
            if request.model:
                # The router will handle model resolution
                pass

            # Run agent
            result = agent.invoke(
                session_id=request_id,
                user_message=user_message,
            )

            return result

        finally:
            agent.close()

    async def run_streaming(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> Any:
        """Run streaming completion.

        Args:
            request: The chat completion request
            request_id: Unique request identifier

        Yields:
            Tuples of (token, metadata) from the agent's streaming response
        """
        # Create fresh agent for this request (no checkpointing for API mode)
        agent = Agent(
            llm=self.llm_router,
            mcp_executor=self.mcp_executor,
            db_path=None,  # Disable checkpointing for API requests
            workspace_root=self.config.workspace_path,
        )

        try:
            # Extract the last user message as the input
            user_message = self._extract_last_user_message(request.messages)

            # Stream agent response
            async for token, metadata in agent.astream(
                session_id=request_id,
                user_message=user_message,
            ):
                yield token, metadata

        finally:
            agent.close()

    def _extract_last_user_message(self, messages: list) -> str:
        """Extract the last user message from the conversation.

        Args:
            messages: List of chat messages

        Returns:
            The content of the last user message
        """
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    def _format_conversation_history(self, messages: list) -> str:
        """Format full conversation history for context.

        Args:
            messages: List of chat messages

        Returns:
            Formatted conversation history string
        """
        lines = []
        for msg in messages:
            lines.append(f"[{msg.role}]: {msg.content}")
        return "\n".join(lines)
