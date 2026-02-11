"""Executor node implementation - executes tasks in the plan."""

import logging

from asterism.agent.nodes.executor.task_runner import create_task_runner
from asterism.agent.nodes.shared import (
    advance_task,
    are_dependencies_satisfied,
    create_error_state,
    get_current_task,
    is_linear_plan,
)
from asterism.agent.state import AgentState
from asterism.llm.providers import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

logger = logging.getLogger(__name__)


def executor_node(
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
    state: AgentState,
) -> AgentState:
    """Execute the current task in the plan.

    For linear plans (sequential tasks with simple dependencies), this will
    batch execute all remaining tasks in a single pass, reducing the number
    of evaluator calls needed.

    Args:
        llm: The LLM provider for LLM-only tasks.
        mcp_executor: The MCP executor for tool calls.
        state: Current agent state.

    Returns:
        Updated state with execution result(s).
    """
    plan = state.get("plan")

    # Check if this is a linear plan that can be batch executed
    if is_linear_plan(plan):
        return _execute_linear_plan(llm, mcp_executor, state)

    # Standard single-task execution for non-linear plans
    return _execute_single_task(llm, mcp_executor, state)


def _execute_linear_plan(
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
    state: AgentState,
) -> AgentState:
    """Execute tasks in a linear plan sequentially without intermediate evaluations.

    This optimization executes all remaining tasks in a linear plan in one pass,
    only stopping if a task fails. This eliminates unnecessary evaluator calls
    between tasks in a simple sequential workflow.

    Args:
        llm: The LLM provider for LLM-only tasks.
        mcp_executor: The MCP executor for tool calls.
        state: Current agent state.

    Returns:
        Updated state with all execution results.
    """
    # plan = state.get("plan")
    current_state = state
    executed_count = 0

    while True:
        task = get_current_task(current_state)

        if not task:
            # No more tasks to execute
            break

        if not are_dependencies_satisfied(task, current_state):
            deps = [d for d in task.depends_on]
            return create_error_state(current_state, f"Dependencies not satisfied: {deps}")

        logger.info(f"[executor] Starting task {task.id}: {task.description[:80]}")

        runner = create_task_runner(task, llm, mcp_executor)
        result = runner.execute(task, current_state)

        log_task_completion(task.id, result.success)
        executed_count += 1

        # Advance to next task
        current_state = advance_task(current_state, result)

        # Stop batch execution if task failed
        if not result.success:
            logger.info("[executor] Stopping batch execution due to task failure")
            break

    if executed_count > 1:
        logger.info(f"[executor] Batch executed {executed_count} tasks in linear plan")

    return current_state


def _execute_single_task(
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
    state: AgentState,
) -> AgentState:
    """Execute a single task (standard mode for non-linear plans).

    Args:
        llm: The LLM provider for LLM-only tasks.
        mcp_executor: The MCP executor for tool calls.
        state: Current agent state.

    Returns:
        Updated state with execution result.
    """
    task = get_current_task(state)

    if not task:
        return create_error_state(state, "No task to execute")

    if not are_dependencies_satisfied(task, state):
        deps = [d for d in task.depends_on]
        return create_error_state(state, f"Dependencies not satisfied: {deps}")

    logger.info(f"[executor] Starting task {task.id}: {task.description[:80]}")

    runner = create_task_runner(task, llm, mcp_executor)
    result = runner.execute(task, state)

    log_task_completion(task.id, result.success)

    return advance_task(state, result)


def log_task_completion(task_id: str, success: bool) -> None:
    """Log task completion status."""
    if success:
        logger.info(f"[executor] Task {task_id} completed successfully")
    else:
        logger.warning(f"[executor] Task {task_id} failed")
