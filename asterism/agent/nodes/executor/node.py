"""Executor node implementation - executes tasks in the plan."""

import logging

from asterism.agent.nodes.executor.task_runner import create_task_runner
from asterism.agent.nodes.shared import (
    advance_task,
    are_dependencies_satisfied,
    create_error_state,
    get_current_task,
)
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

logger = logging.getLogger(__name__)


def executor_node(
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
    state: AgentState,
) -> AgentState:
    """Execute the current task in the plan.

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
