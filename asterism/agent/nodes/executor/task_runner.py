"""Task runner abstractions and factory."""

from typing import Protocol

from asterism.agent.models import TaskResult
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

from .llm_runner import LLMRunner
from .mcp_runner import MCPRunner


class TaskRunner(Protocol):
    """Protocol for task execution strategies."""

    def execute(self, task, state: AgentState) -> TaskResult:
        """Execute a task.

        Args:
            task: The task to execute.
            state: Current agent state.

        Returns:
            TaskResult with execution outcome.
        """
        ...


def create_task_runner(
    task,
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
) -> TaskRunner:
    """Create appropriate runner for the task.

    Args:
        task: The task to create runner for.
        llm: LLM provider for LLM tasks.
        mcp_executor: MCP executor for tool tasks.

    Returns:
        TaskRunner instance appropriate for the task type.
    """
    if task.tool_call:
        return MCPRunner(mcp_executor)
    return LLMRunner(llm)
