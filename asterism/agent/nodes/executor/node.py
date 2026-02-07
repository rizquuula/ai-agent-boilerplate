"""Executor node implementation."""

import logging

from asterism.agent.models import LLMUsage, TaskResult
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

logger = logging.getLogger(__name__)


def executor_node(llm: BaseLLMProvider, mcp_executor: MCPExecutor, state: AgentState) -> AgentState:
    """
    Execute the current task in the plan.

    Args:
        llm: The LLM provider for LLM-only tasks.
        mcp_executor: The MCP executor for tool calls.
        state: Current agent state.

    Returns:
        Updated state with execution result.
    """
    plan = state.get("plan")
    if not plan:
        new_state = state.copy()
        new_state["error"] = "No plan to execute"
        return new_state

    current_index = state.get("current_task_index", 0)

    # Check if all tasks are complete
    if current_index >= len(plan.tasks):
        new_state = state.copy()
        new_state["error"] = "All tasks already completed"
        return new_state

    task = plan.tasks[current_index]

    # Check dependencies
    completed_task_ids = {result.task_id for result in state.get("execution_results", [])}
    unsatisfied_deps = [dep for dep in task.depends_on if dep not in completed_task_ids]
    if unsatisfied_deps:
        new_state = state.copy()
        new_state["error"] = f"Task dependencies not satisfied: {unsatisfied_deps}"
        return new_state

    # Execute the task
    result = TaskResult(task_id=task.id, success=False, result=None, error=None)
    task_usage: LLMUsage | None = None

    logger.info(f"Executing task {task.id}: {task.description[:100]}")

    try:
        if task.tool_call:
            # Parse tool call: "server_name:tool_name"
            if ":" not in task.tool_call:
                raise ValueError(f"Invalid tool_call format: {task.tool_call}. Expected 'server:tool'")

            server_name, tool_name = task.tool_call.split(":", 1)
            tool_input = task.tool_input or {}

            logger.debug(f"Tool call: {server_name}:{tool_name}, input: {tool_input}")

            # Execute via MCP
            mcp_result = mcp_executor.execute_tool(server_name, tool_name, **tool_input)

            if mcp_result["success"]:
                result.success = True
                result.result = mcp_result["result"]
                logger.info(f"Task {task.id} succeeded")
            else:
                result.error = mcp_result["error"]
                logger.warning(f"Task {task.id} failed: {result.error}")
        else:
            # LLM-only task
            if not task.description:
                raise ValueError("Task description is required for LLM-only tasks")

            # Build context from dependent task results
            execution_context = ""
            if task.depends_on:
                execution_results = state.get("execution_results", [])
                dependent_results = [r for r in execution_results if r.task_id in task.depends_on]
                if dependent_results:
                    execution_context = "\n\nContext from previous tasks:\n"
                    for result in dependent_results:
                        if result.success:
                            execution_context += f"\n--- Result from task '{result.task_id}' ---\n{result.result}\n"
                        else:
                            execution_context += f"\n--- Task '{result.task_id}' failed: {result.error}\n"

            # Build full prompt with context
            full_prompt = task.description
            if execution_context:
                full_prompt = f"{task.description}\n\n{execution_context}"

            # Use LLM to process with usage tracking
            llm_response = llm.invoke_with_usage(full_prompt)
            result.success = True
            result.result = llm_response.content

            # Track LLM usage for this task
            task_usage = LLMUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                model=llm.model,
                node_name="executor_node",
            )
            result.llm_usage = task_usage

    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.error(f"Task {task.id} execution error: {e}")

    # Update state
    new_state = state.copy()
    new_state["execution_results"] = state.get("execution_results", []) + [result]
    new_state["current_task_index"] = current_index + 1
    new_state["error"] = None if result.success else result.error

    # Add LLM usage to state if this was an LLM task
    if task_usage:
        new_state["llm_usage"] = state.get("llm_usage", []) + [task_usage]

    return new_state
