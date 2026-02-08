"""Executor node implementation."""

import logging
import time

from asterism.agent.models import LLMUsage, TaskResult
from asterism.agent.state import AgentState
from asterism.agent.utils import log_llm_call, log_mcp_tool_call, log_task_execution
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
    task_start_time = time.perf_counter()

    logger.info(f"[executor] Starting task {task.id}: {task.description[:100]}")

    try:
        if task.tool_call:
            # Parse tool call: "server_name:tool_name"
            if ":" not in task.tool_call:
                raise ValueError(f"Invalid tool_call format: {task.tool_call}. Expected 'server:tool'")

            server_name, tool_name = task.tool_call.split(":", 1)
            tool_input = task.tool_input or {}

            logger.debug(f"[executor] MCP tool call: {server_name}:{tool_name}, input_keys: {list(tool_input.keys())}")

            # Execute via MCP with timing
            mcp_start_time = time.perf_counter()
            mcp_result = mcp_executor.execute_tool(server_name, tool_name, **tool_input)
            mcp_duration_ms = (time.perf_counter() - mcp_start_time) * 1000

            if mcp_result["success"]:
                result.success = True
                result.result = mcp_result["result"]

                # Log successful MCP tool call
                log_mcp_tool_call(
                    logger=logger,
                    server_name=server_name,
                    tool_name=tool_name,
                    input_keys=list(tool_input.keys()),
                    success=True,
                    duration_ms=mcp_duration_ms,
                    result_preview=str(mcp_result.get("result", ""))[:500],
                )
            else:
                result.error = mcp_result["error"]

                # Log failed MCP tool call
                log_mcp_tool_call(
                    logger=logger,
                    server_name=server_name,
                    tool_name=tool_name,
                    input_keys=list(tool_input.keys()),
                    success=False,
                    duration_ms=mcp_duration_ms,
                    error=mcp_result.get("error"),
                )
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
                    for dep_result in dependent_results:
                        if dep_result.success:
                            execution_context += (
                                f"\n--- Result from task '{dep_result.task_id}' ---\n{dep_result.result}\n"
                            )
                        else:
                            execution_context += f"\n--- Task '{dep_result.task_id}' failed: {dep_result.error}\n"

            # Build full prompt with context
            full_prompt = task.description
            if execution_context:
                full_prompt = f"{task.description}\n\n{execution_context}"

            # Use LLM to process with usage tracking and timing
            llm_start_time = time.perf_counter()
            llm_response = llm.invoke_with_usage(full_prompt)
            llm_duration_ms = (time.perf_counter() - llm_start_time) * 1000

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

            # Log LLM task execution
            log_llm_call(
                logger=logger,
                node_name="executor_node",
                model=llm.model,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                duration_ms=llm_duration_ms,
                prompt_preview=full_prompt[:500],
                response_preview=llm_response.content[:500],
                success=True,
            )

        # Calculate total task duration
        task_duration_ms = (time.perf_counter() - task_start_time) * 1000

        # Log task completion
        log_task_execution(
            logger=logger,
            task_id=task.id,
            task_type="tool" if task.tool_call else "llm",
            success=result.success,
            duration_ms=task_duration_ms,
            tool_call=task.tool_call,
            error=result.error,
            result_preview=str(result.result)[:500] if result.result else None,
        )

    except Exception as e:
        task_duration_ms = (time.perf_counter() - task_start_time) * 1000
        result.error = str(e)
        result.success = False

        # Log task failure
        log_task_execution(
            logger=logger,
            task_id=task.id,
            task_type="tool" if task.tool_call else "llm",
            success=False,
            duration_ms=task_duration_ms,
            tool_call=task.tool_call,
            error=str(e),
        )

        logger.error(f"[executor] Task {task.id} execution error: {e}", exc_info=True)

    # Update state
    new_state = state.copy()
    new_state["execution_results"] = state.get("execution_results", []) + [result]
    new_state["current_task_index"] = current_index + 1
    new_state["error"] = None if result.success else result.error

    # Add LLM usage to state if this was an LLM task
    if task_usage:
        new_state["llm_usage"] = state.get("llm_usage", []) + [task_usage]

    return new_state
