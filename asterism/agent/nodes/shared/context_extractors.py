"""Context extraction utilities from agent state.

These functions extract and format information from agent state
for use in prompts and business logic.
"""

from langchain_core.messages import HumanMessage

from asterism.agent.models import TaskResult
from asterism.agent.state import AgentState


def get_user_request(state: AgentState) -> str:
    """Extract the original user request from state messages.

    Finds the first human message in the conversation history.

    Args:
        state: Current agent state.

    Returns:
        The user's original request or a default message.
    """
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return "No user request found"


def get_last_result(state: AgentState) -> TaskResult | None:
    """Get the most recent execution result.

    Args:
        state: Current agent state.

    Returns:
        The last task result, or None if no results exist.
    """
    results = state.get("execution_results", [])
    return results[-1] if results else None


def get_current_task(state: AgentState):
    """Get the current task from the plan.

    Args:
        state: Current agent state.

    Returns:
        The current task object, or None if no plan or all tasks completed.
    """
    plan = state.get("plan")
    if not plan:
        return None

    current_index = state.get("current_task_index", 0)
    if current_index >= len(plan.tasks):
        return None

    return plan.tasks[current_index]


def format_execution_history(results: list[TaskResult], max_results: int = 10) -> str:
    """Format execution results as readable history.

    Args:
        results: List of task results.
        max_results: Maximum number of results to include.

    Returns:
        Formatted string of execution history.
    """
    if not results:
        return "No tasks executed yet."

    lines = []
    for result in results[-max_results:]:  # Show most recent
        status = "✓" if result.success else "✗"
        if result.success:
            result_str = str(result.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "... [truncated]"
            lines.append(f"{status} {result.task_id}: {result_str}")
        else:
            lines.append(f"{status} {result.task_id}: ERROR - {result.error}")

    return "\n".join(lines)


def format_execution_summary(state: AgentState) -> str:
    """Format a summary of execution progress.

    Args:
        state: Current agent state.

    Returns:
        Formatted summary of completed tasks.
    """
    results = state.get("execution_results", [])
    if not results:
        return "No execution history."

    completed = len([r for r in results if r.success])
    failed = len([r for r in results if not r.success])

    return f"Completed: {completed}, Failed: {failed}, Total: {len(results)}"


def get_completed_task_ids(state: AgentState) -> set[str]:
    """Get set of completed task IDs.

    Args:
        state: Current agent state.

    Returns:
        Set of task IDs that have been executed.
    """
    results = state.get("execution_results", [])
    return {result.task_id for result in results}


def are_dependencies_satisfied(task, state: AgentState) -> bool:
    """Check if all dependencies for a task are satisfied.

    Args:
        task: The task to check.
        state: Current agent state.

    Returns:
        True if all dependencies are satisfied, False otherwise.
    """
    if not task.depends_on:
        return True

    completed_ids = get_completed_task_ids(state)
    return all(dep in completed_ids for dep in task.depends_on)


def get_failed_tasks(state: AgentState) -> list[TaskResult]:
    """Get list of failed task results.

    Args:
        state: Current agent state.

    Returns:
        List of failed task results.
    """
    results = state.get("execution_results", [])
    return [r for r in results if not r.success]


def has_execution_history(state: AgentState) -> bool:
    """Check if there's any execution history.

    Args:
        state: Current agent state.

    Returns:
        True if there are execution results.
    """
    return bool(state.get("execution_results"))
