"""Agent lifecycle management service."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

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
            # Convert OpenAI format messages to LangChain messages
            messages = self._convert_messages(request.messages)

            # Set model if specified in request
            if request.model:
                # The router will handle model resolution
                pass

            # Run agent with full conversation context
            result = agent.invoke(
                session_id=request_id,
                messages=messages,
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
            # Convert OpenAI format messages to LangChain messages
            messages = self._convert_messages(request.messages)

            # Stream agent response with full conversation context
            async for token, metadata in agent.astream(
                session_id=request_id,
                messages=messages,
            ):
                yield token, metadata

        finally:
            agent.close()

    def _convert_messages(self, messages: list) -> list:
        """Convert OpenAI format messages to LangChain messages.

        Args:
            messages: List of ChatMessage objects with role and content

        Returns:
            List of LangChain BaseMessage objects
        """
        converted = []
        for msg in messages:
            if msg.role == "system":
                converted.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                converted.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                converted.append(AIMessage(content=msg.content))
            elif msg.role == "tool":
                # Tool messages require tool_call_id to link to the assistant's tool call
                converted.append(
                    ToolMessage(
                        content=msg.content,
                        tool_call_id=msg.tool_call_id or "",
                        name=msg.name or "tool",
                    )
                )
        return converted

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
