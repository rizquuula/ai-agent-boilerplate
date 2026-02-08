"""Planner node implementation."""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import LLMUsage, Plan
from asterism.agent.state import AgentState
from asterism.agent.utils import log_llm_call, log_plan_created
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

from .prompts import PLANNER_SYSTEM_PROMPT
from .utils import format_tools_context, generate_task_id

logger = logging.getLogger(__name__)


def planner_node(llm: BaseLLMProvider, mcp_executor: MCPExecutor, state: AgentState) -> AgentState:
    """
    Create or update a plan based on the user request and execution history.

    The LLM will receive:
    1. SOUL.md + AGENT.md as a SystemMessage (loaded fresh from disk if configured)
    2. Node-specific planning instructions as a SystemMessage (including available tools)
    3. User request and execution context as a HumanMessage

    Args:
        llm: The LLM provider to use for planning.
        mcp_executor: The MCP executor for discovering available tools.
        state: Current agent state.

    Returns:
        Updated state with a new or updated plan.
    """
    # Get available MCP tools
    try:
        tool_schemas = mcp_executor.get_tool_schemas()
        tools_context = format_tools_context(tool_schemas)
    except Exception as e:
        tools_context = f"Error getting tool information: {str(e)}"

    # Build context from messages
    user_message = ""
    if state["messages"]:
        # Get the latest user message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

    execution_context = ""
    if state["execution_results"]:
        execution_context = "\n\nExecution History:\n"
        for result in state["execution_results"]:
            status = "✓" if result.success else "✗"
            execution_context += f"- {status} {result.task_id}: {result.result if result.success else result.error}\n"

    # Enhanced system prompt with tool information
    enhanced_system_prompt = f"""{PLANNER_SYSTEM_PROMPT}

Available MCP Tools:
{tools_context}

When creating tasks with tool calls:
- Use ONLY tools listed above
- Use exact format: "server_name:tool_name"
- Provide all required parameters in tool_input
- If a tool is not available, use LLM reasoning instead
"""

    user_prompt = f"""User Request: {user_message}

{execution_context if execution_context else ""}

Create a plan to accomplish this request using the available tools.

JSON OUTPUT:"""

    try:
        # Use structured output with message list
        # The provider will auto-prepend SOUL.md + AGENT.md as a SystemMessage
        messages = [
            SystemMessage(content=enhanced_system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Log prompt preview
        prompt_preview = f"System: {enhanced_system_prompt[:200]}... User: {user_prompt[:200]}..."
        logger.debug(f"[planner] Sending LLM request with prompt preview: {prompt_preview[:300]}")

        # Time the LLM call
        start_time = time.perf_counter()
        response = llm.invoke_structured(messages, Plan)
        duration_ms = (time.perf_counter() - start_time) * 1000

        plan = response.parsed

        # Log LLM call with structured context
        log_llm_call(
            logger=logger,
            node_name="planner_node",
            model=llm.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            duration_ms=duration_ms,
            prompt_preview=prompt_preview,
            response_preview=str(plan.model_dump())[:500] if plan else None,
            success=True,
        )

        # Validate the plan
        if not plan or not plan.tasks:
            raise ValueError("Plan has no tasks")

        # Ensure all tasks have IDs
        for i, task in enumerate(plan.tasks):
            if not task.id:
                task.id = generate_task_id(i, task.description)

        # Log plan creation with structured context
        has_dependencies = any(t.depends_on for t in plan.tasks)
        log_plan_created(
            logger=logger,
            task_count=len(plan.tasks),
            task_ids=[t.id for t in plan.tasks],
            has_dependencies=has_dependencies,
            reasoning_preview=plan.reasoning,
        )

        # Update state
        new_state = state.copy()
        new_state["plan"] = plan
        new_state["current_task_index"] = 0
        new_state["error"] = None

        # Track LLM usage from structured output response
        usage = LLMUsage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            model=llm.model,
            node_name="planner_node",
        )
        new_state["llm_usage"] = state.get("llm_usage", []) + [usage]

        return new_state

    except Exception as e:
        error_msg = f"Planning failed: {str(e)}"
        duration_ms = (time.perf_counter() - start_time) * 1000 if "start_time" in locals() else 0

        # Log failed LLM call if we got that far
        if "response" in locals():
            log_llm_call(
                logger=logger,
                node_name="planner_node",
                model=llm.model,
                prompt_tokens=getattr(response, "prompt_tokens", 0),
                completion_tokens=getattr(response, "completion_tokens", 0),
                duration_ms=duration_ms,
                prompt_preview=prompt_preview if "prompt_preview" in locals() else None,
                success=False,
                error=str(e),
            )

        logger.error(f"[planner] Planning failed: {error_msg}", exc_info=True)

        new_state = state.copy()
        new_state["error"] = error_msg
        # Add the error to messages for context in case of replanning
        new_state["messages"] = state.get("messages", []) + [HumanMessage(content=f"[Planning Error] {error_msg}")]
        return new_state
