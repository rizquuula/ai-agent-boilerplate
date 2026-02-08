"""Planner node implementation - creates execution plans."""

import logging

from asterism.agent.models import Plan
from asterism.agent.nodes.planner.context import build_planner_context
from asterism.agent.nodes.planner.service import (
    PlanningError,
    log_plan_creation,
    validate_and_enrich_plan,
)
from asterism.agent.nodes.shared import (
    LLMCaller,
    create_error_state,
    set_plan,
)
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor

logger = logging.getLogger(__name__)


def planner_node(
    llm: BaseLLMProvider,
    mcp_executor: MCPExecutor,
    state: AgentState,
    workspace_root: str = "./workspace",
) -> AgentState:
    """Create or update a plan based on user request and execution history.

    Args:
        llm: The LLM provider for planning.
        mcp_executor: The MCP executor for tool discovery.
        state: Current agent state.
        workspace_root: Path to workspace for context.

    Returns:
        Updated state with new plan.
    """
    logger.info("[planner] Starting plan creation")

    context = build_planner_context(state, mcp_executor, workspace_root)
    caller = LLMCaller(llm, "planner_node")

    try:
        result = caller.call_structured(context.messages, Plan, "creating plan")
        plan = validate_and_enrich_plan(result.parsed)
        log_plan_creation(plan)

        logger.info(f"[planner] Created plan with {len(plan.tasks)} tasks")
        return set_plan(state, plan, result.usage)

    except PlanningError as e:
        logger.error(f"[planner] Plan validation failed: {e}")
        return create_error_state(state, f"Planning failed: {e}")

    except Exception as e:
        logger.error(f"[planner] Planning failed: {e}", exc_info=True)
        return create_error_state(state, f"Planning failed: {e}")
