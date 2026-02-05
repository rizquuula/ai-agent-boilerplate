"""Evaluator node for deciding next steps after task execution."""

from typing import Literal

from asterism.agent.state import AgentState


def evaluator_node(state: AgentState) -> AgentState:
    """
    Evaluate execution results and decide next action.

    Args:
        state: Current agent state.

    Returns:
        Updated state (routing is handled by should_continue function).
    """
    # This node doesn't modify state, it just evaluates
    # The routing function determines the next step
    return state


def should_continue(state: AgentState) -> Literal["planner_node", "executor_node", "finalizer_node"]:
    """
    Routing function to determine next node after evaluation.

    Args:
        state: Current agent state.

    Returns:
        Name of the next node to execute.
    """
    # Check for errors or need to re-plan
    if state.get("error"):
        return "planner_node"

    plan = state.get("plan")
    if not plan:
        return "planner_node"

    current_index = state.get("current_task_index", 0)
    total_tasks = len(plan.tasks)

    # Check if all tasks are complete
    if current_index >= total_tasks:
        return "finalizer_node"

    # Check if the last task failed
    if state.get("execution_results"):
        last_result = state["execution_results"][-1]
        if not last_result.success:
            return "planner_node"

    # Continue executing
    return "executor_node"
