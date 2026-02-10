"""Main Agent implementation using LangGraph."""

import sqlite3
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from asterism.agent.graph_builders import build_full_graph, build_streaming_graph
from asterism.agent.models import AgentResponse
from asterism.agent.nodes.finalizer.prompts import FINALIZER_SYSTEM_PROMPT
from asterism.agent.nodes.shared import build_execution_trace, get_user_request
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor


def _initialize_state(session_id: str, messages: list[BaseMessage]) -> AgentState:
    """Create initial agent state."""
    return {
        "session_id": session_id,
        "trace_id": str(uuid.uuid4()),  # Generate unique trace ID for this flow
        "messages": messages,
        "plan": None,
        "current_task_index": 0,
        "execution_results": [],
        "final_response": None,
        "error": None,
        "llm_usage": [],
    }


class Agent:
    """An Agent can do plan, execute, and manage tasks using LangGraph."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        mcp_executor: MCPExecutor,
        db_path: str | None = None,
        workspace_root: str = "./workspace",
    ):
        """
        Initialize the agent.

        Args:
            llm: LLM provider for planning and response generation.
            mcp_executor: MCP executor for tool calls.
            db_path: Path to SQLite database for checkpoint storage. If None, uses default.
            workspace_root: Path to the workspace directory for context generation (default: ./workspace).
        """
        self.llm = llm
        self.mcp_executor = mcp_executor
        self.db_path = db_path  # Allow None for stateless mode
        self.workspace_root = workspace_root
        self._full_graph = None
        self._streaming_graph = None
        self._checkpointer: BaseCheckpointSaver | None = None
        self._conn: sqlite3.Connection | None = None

    def _get_checkpointer(self) -> SqliteSaver | None:
        """Get or create the SQLite checkpointer.

        Returns:
            SqliteSaver instance or None if db_path is None (stateless mode)
        """
        if self.db_path is None:
            # Stateless mode - no checkpointing
            return None

        if self._checkpointer is None:
            from pathlib import Path

            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Use check_same_thread=False for multi-threaded environments
            if self.db_path == ":memory:":
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            else:
                self._conn = sqlite3.connect(self.db_path)
            self._checkpointer = SqliteSaver(self._conn)
        return self._checkpointer

    def build(self):
        """Build the full LangGraph workflow (with finalizer).

        Returns:
            Compiled StateGraph ready for execution.
        """
        if self._full_graph is not None:
            return self._full_graph

        checkpointer = self._get_checkpointer()
        self._full_graph = build_full_graph(self, checkpointer)
        return self._full_graph

    def build_for_streaming(self):
        """Build the streaming LangGraph workflow (stops before finalizer).

        Returns:
            Compiled StateGraph ready for execution.
        """
        if self._streaming_graph is not None:
            return self._streaming_graph

        checkpointer = self._get_checkpointer()
        self._streaming_graph = build_streaming_graph(self, checkpointer)
        return self._streaming_graph

    def invoke(self, session_id: str, messages: list[BaseMessage]) -> dict[str, Any]:
        """Process messages and return the agent's response.

        Args:
            session_id: Unique session identifier for state persistence.
            messages: List of messages (system, user, assistant, tool) in the conversation.

        Returns:
            Dictionary containing:
                - message: Natural language response
                - execution_trace: List of execution steps
                - plan_used: The plan that was executed (if any)
                - session_id: The session ID used
        """
        # Build graph if needed
        graph = self.build()

        # Get initial state
        initial_state = _initialize_state(session_id, messages)

        # Run the graph
        try:
            final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": session_id}})
        except Exception as e:
            # Graph execution failed
            return {
                "message": f"Agent execution failed: {str(e)}",
                "execution_trace": [],
                "plan_used": None,
                "session_id": session_id,
                "error": str(e),
            }

        # Extract response
        response: AgentResponse | None = final_state.get("final_response")

        if response is None:
            return {
                "message": "Agent did not produce a response",
                "execution_trace": [],
                "plan_used": None,
                "session_id": session_id,
                "error": "No final response generated",
            }

        # Aggregate LLM usage from all nodes
        llm_usage_list = final_state.get("llm_usage", [])
        total_usage = {
            "total_prompt_tokens": sum(u.prompt_tokens for u in llm_usage_list),
            "total_completion_tokens": sum(u.completion_tokens for u in llm_usage_list),
            "total_tokens": sum(u.total_tokens for u in llm_usage_list),
            "calls_by_node": {},
        }
        for usage in llm_usage_list:
            node = usage.node_name
            total_usage["calls_by_node"][node] = total_usage["calls_by_node"].get(node, 0) + 1

        return {
            "message": response.message,
            "execution_trace": response.execution_trace,
            "plan_used": response.plan_used.model_dump() if response.plan_used else None,
            "session_id": session_id,
            "total_usage": total_usage,
        }

    async def astream(
        self, session_id: str, messages: list[BaseMessage]
    ) -> AsyncGenerator[tuple[str, dict[str, Any] | None]]:
        """Process messages and stream the agent's response tokens.

        This method runs planner → executor → evaluator, then stops before
        the finalizer. It then streams the final response tokens manually.

        Args:
            session_id: Unique session identifier.
            messages: List of messages (system, user, assistant, tool) in the conversation.

        Yields:
            Tuples of (token, metadata) where:
                - token: A string token from the streaming response
                - metadata: Dict with execution info (only on final token, None otherwise)

        Example:
            async for token, metadata in agent.astream("session-1", messages):
                if metadata:
                    print(f"Complete! Usage: {metadata['total_usage']}")
                else:
                    print(token, end="")
        """
        from langchain_core.messages import SystemMessage

        # Build streaming graph (stops before finalizer)
        graph = self.build_for_streaming()

        # Get initial state
        initial_state = _initialize_state(session_id, messages)

        # Run the graph up to finalization (non-streaming for planning/execution)
        try:
            final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": session_id}})
        except Exception as e:
            # Graph execution failed
            yield (
                f"Agent execution failed: {str(e)}",
                {
                    "error": str(e),
                    "session_id": session_id,
                    "execution_trace": [],
                    "plan_used": None,
                    "total_usage": {},
                },
            )
            return

        # Check for errors in execution
        error = final_state.get("error")
        if error:
            yield (
                f"Error: {error}",
                {
                    "error": error,
                    "session_id": session_id,
                    "execution_trace": [],
                    "plan_used": None,
                    "total_usage": {},
                },
            )
            return

        # Build execution trace for metadata
        trace = build_execution_trace(final_state)
        plan_used = final_state.get("plan")

        # Aggregate LLM usage from planning/execution nodes
        llm_usage_list = final_state.get("llm_usage", [])
        total_usage = {
            "total_prompt_tokens": sum(u.prompt_tokens for u in llm_usage_list),
            "total_completion_tokens": sum(u.completion_tokens for u in llm_usage_list),
            "total_tokens": sum(u.total_tokens for u in llm_usage_list),
            "calls_by_node": {},
        }
        for usage in llm_usage_list:
            node = usage.node_name
            total_usage["calls_by_node"][node] = total_usage["calls_by_node"].get(node, 0) + 1

        # Now stream the final response using the LLM's astream
        user_request = get_user_request(final_state)

        # Format execution results as summary
        execution_results = final_state.get("execution_results", [])
        if execution_results:
            results_summary = "\n".join(f"Task {r.task_id}: {r.result}" for r in execution_results)
        else:
            results_summary = "No execution results."

        user_prompt = f"""Original user request: {user_request}

Execution results:
{results_summary}

Create a response for the user."""

        finalizer_messages = [
            SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Stream tokens from the LLM
        full_response = ""
        try:
            async for token in self.llm.astream(finalizer_messages):
                full_response += token
                yield token, None  # Yield token with no metadata during streaming

            # Final yield with metadata
            metadata = {
                "session_id": session_id,
                "execution_trace": trace,
                "plan_used": plan_used.model_dump() if plan_used else None,
                "total_usage": total_usage,
                "message": full_response,
            }
            yield "", metadata

        except Exception as e:
            # If streaming fails, yield error
            yield (
                f"\n[Streaming failed: {str(e)}]",
                {
                    "error": str(e),
                    "session_id": session_id,
                    "execution_trace": trace,
                    "plan_used": plan_used.model_dump() if plan_used else None,
                    "total_usage": total_usage,
                    "message": full_response,
                },
            )

    def clear_session(self, session_id: str) -> None:
        """Clear all state for a session.

        Args:
            session_id: The session ID to clear.
        """
        if self.db_path is None:
            # Stateless mode - no session to clear
            return

        if self._checkpointer is None:
            self.build()

        if self._checkpointer is None:
            return

        # Delete all checkpoints for this session
        conn = self._checkpointer.conn
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (session_id,),
            )
            conn.commit()
        finally:
            cur.close()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
