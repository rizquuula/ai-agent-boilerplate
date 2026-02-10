"""Main Agent implementation using LangGraph."""

import sqlite3
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from asterism.agent.models import AgentResponse
from asterism.agent.nodes import evaluator_node, executor_node, finalizer_node, planner_node, should_continue
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor


def _initialize_state(session_id: str, user_message: str) -> AgentState:
    """Create initial agent state."""
    return {
        "session_id": session_id,
        "trace_id": str(uuid.uuid4()),  # Generate unique trace ID for this flow
        "messages": [HumanMessage(content=user_message)],
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
        self._graph = None
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

    def build(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph ready for execution.
        """
        if self._graph is not None:
            return self._graph

        # Create state graph
        workflow = StateGraph(AgentState)

        # Add nodes with dependencies injected via closures
        workflow.add_node("planner_node", self._make_planner_node())
        workflow.add_node("executor_node", self._make_executor_node())
        workflow.add_node("evaluator_node", self._make_evaluator_node())
        workflow.add_node("finalizer_node", self._make_finalizer_node())

        # Define edges
        workflow.add_edge(START, "planner_node")
        workflow.add_edge("planner_node", "executor_node")
        workflow.add_edge("executor_node", "evaluator_node")
        workflow.add_conditional_edges(
            "evaluator_node",
            self._make_routing_function(),
            {
                "planner_node": "planner_node",
                "executor_node": "executor_node",
                "finalizer_node": "finalizer_node",
            },
        )
        workflow.add_edge("finalizer_node", END)

        # Compile with or without checkpointing
        checkpointer = self._get_checkpointer()
        if checkpointer:
            self._graph = workflow.compile(checkpointer=checkpointer)
        else:
            # Stateless mode - no checkpointing
            self._graph = workflow.compile()

        return self._graph

    def _make_planner_node(self):
        """Create planner node with LLM and MCP dependencies."""
        llm = self.llm
        mcp_executor = self.mcp_executor
        workspace_root = self.workspace_root

        def _planner_node(state: AgentState) -> AgentState:
            return planner_node(llm, mcp_executor, state, workspace_root)

        return _planner_node

    def _make_executor_node(self):
        """Create executor node with LLM and MCP dependencies."""
        llm = self.llm
        mcp_executor = self.mcp_executor

        def _executor_node(state: AgentState) -> AgentState:
            return executor_node(llm, mcp_executor, state)

        return _executor_node

    def _make_evaluator_node(self):
        """Create evaluator node with LLM dependency."""
        llm = self.llm

        def _evaluator_node(state: AgentState) -> AgentState:
            return evaluator_node(llm, state)

        return _evaluator_node

    def _make_finalizer_node(self):
        """Create finalizer node with LLM dependency."""
        llm = self.llm

        def _finalizer_node(state: AgentState) -> AgentState:
            return finalizer_node(llm, state)

        return _finalizer_node

    def _make_routing_function(self):
        """Create routing function for evaluator."""

        def _route(state: AgentState) -> str:
            return should_continue(state)

        return _route

    def invoke(self, session_id: str, user_message: str) -> dict[str, Any]:
        """
        Process a user message and return the agent's response.

        Args:
            session_id: Unique session identifier for state persistence.
            user_message: The user's input message.

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
        initial_state = _initialize_state(session_id, user_message)

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

    async def astream(self, session_id: str, user_message: str) -> AsyncGenerator[tuple[str, dict[str, Any] | None]]:
        """
        Process a user message and stream the agent's response tokens.

        This method runs planner → executor → evaluator normally, then streams
        the final response tokens from the finalizer.

        Args:
            session_id: Unique session identifier.
            user_message: The user's input message.

        Yields:
            Tuples of (token, metadata) where:
                - token: A string token from the streaming response
                - metadata: Dict with execution info (only on final token, None otherwise)

        Example:
            async for token, metadata in agent.astream("session-1", "Hello"):
                if metadata:
                    print(f"Complete! Usage: {metadata['total_usage']}")
                else:
                    print(token, end="")
        """
        from langchain_core.messages import SystemMessage

        from asterism.agent.nodes.finalizer.prompts import FINALIZER_SYSTEM_PROMPT
        from asterism.agent.nodes.shared import (
            build_execution_trace,
            get_user_request,
        )

        # Build graph if needed
        graph = self.build()

        # Get initial state
        initial_state = _initialize_state(session_id, user_message)

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

        messages = [
            SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Stream tokens from the LLM
        full_response = ""
        try:
            async for token in self.llm.astream(messages):
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
        """
        Clear all state for a session.

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
