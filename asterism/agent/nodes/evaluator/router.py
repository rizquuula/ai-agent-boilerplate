"""Routing logic for evaluator decisions."""

from enum import StrEnum

from asterism.agent.models import EvaluationDecision
from asterism.agent.nodes.shared import is_linear_plan
from asterism.agent.state import AgentState


class RouteTarget(StrEnum):
    """Possible routing targets after evaluation."""

    PLANNER = "planner_node"
    EXECUTOR = "executor_node"
    FINALIZER = "finalizer_node"


def determine_route(state: AgentState) -> RouteTarget:
    """Determine next node based on state.

    Uses evaluation_result if available, falls back to logic-based routing.

    For linear plans with all tasks completed successfully, routes directly
    to finalizer without requiring an LLM evaluation, saving tokens and time.

    Args:
        state: Current agent state.

    Returns:
        Target node name for next step.
    """
    # Check for explicit error state first
    if state.get("error"):
        return RouteTarget.PLANNER

    # Use LLM evaluation result if available
    evaluation = state.get("evaluation_result")
    if evaluation:
        return _route_from_decision(evaluation.decision)

    # Fallback logic if no evaluation result
    return _determine_fallback_route(state)


def can_skip_evaluation(state: AgentState) -> bool:
    """Check if LLM evaluation can be skipped for this state.

    For linear plans where all tasks have completed successfully,
    we can skip the evaluator LLM call and go directly to finalizer.

    Args:
        state: Current agent state.

    Returns:
        True if evaluation can be skipped, False otherwise.
    """
    plan = state.get("plan")
    if not plan:
        return False

    # Only skip for linear plans
    if not is_linear_plan(plan):
        return False

    current_index = state.get("current_task_index", 0)
    total_tasks = len(plan.tasks)

    # All tasks completed
    if current_index < total_tasks:
        return False

    # Check all tasks succeeded
    execution_results = state.get("execution_results", [])
    if not execution_results or len(execution_results) < total_tasks:
        return False

    # Verify all results are successful
    for result in execution_results:
        if not result.success:
            return False

    return True


def _route_from_decision(decision: EvaluationDecision) -> RouteTarget:
    """Convert evaluation decision to route target."""
    mapping = {
        EvaluationDecision.REPLAN: RouteTarget.PLANNER,
        EvaluationDecision.CONTINUE: RouteTarget.EXECUTOR,
        EvaluationDecision.FINALIZE: RouteTarget.FINALIZER,
    }
    return mapping.get(decision, RouteTarget.EXECUTOR)


def _determine_fallback_route(state: AgentState) -> RouteTarget:
    """Fallback routing when no evaluation result available.

    Args:
        state: Current agent state.

    Returns:
        Target node based on simple logic.
    """
    plan = state.get("plan")
    if not plan:
        return RouteTarget.PLANNER

    current_index = state.get("current_task_index", 0)
    total_tasks = len(plan.tasks)

    if current_index >= total_tasks:
        return RouteTarget.FINALIZER

    execution_results = state.get("execution_results", [])
    if execution_results and not execution_results[-1].success:
        return RouteTarget.PLANNER

    return RouteTarget.EXECUTOR


def should_continue(state: AgentState) -> str:
    """Compatibility wrapper returning string for LangGraph.

    Args:
        state: Current agent state.

    Returns:
        Target node name as string.
    """
    return str(determine_route(state))
