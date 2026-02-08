"""Evaluator business logic."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import EvaluationDecision, EvaluationResult, LLMUsage
from asterism.agent.nodes.evaluator.prompt_builder import build_evaluator_prompt
from asterism.agent.nodes.evaluator.prompts import EVALUATOR_SYSTEM_PROMPT
from asterism.agent.nodes.evaluator.task_resolver import resolve_next_task_inputs
from asterism.agent.nodes.shared import (
    LLMCaller,
    get_current_task,
    prepare_replan_state,
    set_evaluation_result,
)
from asterism.agent.state import AgentState
from asterism.agent.utils import log_evaluation_decision
from asterism.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def evaluate_with_llm(llm: BaseLLMProvider, state: AgentState) -> tuple[EvaluationResult, LLMUsage]:
    """Perform evaluation using LLM.

    Args:
        llm: The LLM provider for evaluation.
        state: Current agent state.

    Returns:
        Tuple of (EvaluationResult, LLMUsage).

    Raises:
        Exception: If LLM call fails.
    """
    caller = LLMCaller(llm, "evaluator_node")
    prompt = build_evaluator_prompt(state)

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    result = caller.call_structured(messages, EvaluationResult, "evaluating execution progress")

    log_evaluation_decision(
        logger=logger,
        decision=result.parsed.decision,
        reasoning_preview=result.parsed.reasoning,
        suggested_changes=result.parsed.suggested_changes,
    )

    return result.parsed, result.usage


def apply_evaluation_result(
    state: AgentState,
    evaluation: EvaluationResult,
    usage: LLMUsage,
    llm: BaseLLMProvider,
) -> AgentState:
    """Apply evaluation result to state, handling all decision paths.

    Args:
        state: Current agent state.
        evaluation: The evaluation result.
        usage: LLM usage for tracking.
        llm: LLM provider for task resolution if needed.

    Returns:
        Updated state.
    """
    new_state = set_evaluation_result(state, evaluation, usage)

    if evaluation.decision == EvaluationDecision.REPLAN:
        return prepare_replan_state(new_state, evaluation)

    if evaluation.decision == EvaluationDecision.CONTINUE:
        return _handle_continue_decision(new_state, llm)

    return new_state


def _handle_continue_decision(state: AgentState, llm: BaseLLMProvider) -> AgentState:
    """Handle CONTINUE decision by resolving next task inputs if needed.

    Args:
        state: Current agent state.
        llm: LLM provider for task resolution.

    Returns:
        Updated state.
    """
    next_task = get_current_task(state)
    if not next_task or not next_task.tool_call:
        return state

    logger.debug(f"Resolving inputs for task: {next_task.id}")

    resolved_input, resolver_usage = resolve_next_task_inputs(llm, next_task, state)

    if resolved_input is not None:
        next_task.tool_input = resolved_input
        logger.info(f"Resolved inputs for task {next_task.id}: {resolved_input}")

    if resolver_usage:
        # Track resolver LLM usage
        state["llm_usage"] = state.get("llm_usage", []) + [resolver_usage]

    return state


def create_fallback_evaluation(state: AgentState, error: str) -> EvaluationResult:
    """Create fallback evaluation when LLM fails.

    Args:
        state: Current agent state.
        error: The error that occurred.

    Returns:
        Fallback evaluation result.
    """
    from asterism.agent.nodes.evaluator.router import RouteTarget, _determine_fallback_route

    route = _determine_fallback_route(state)
    decision_map = {
        RouteTarget.PLANNER: EvaluationDecision.REPLAN,
        RouteTarget.EXECUTOR: EvaluationDecision.CONTINUE,
        RouteTarget.FINALIZER: EvaluationDecision.FINALIZE,
    }

    decision = decision_map.get(route, EvaluationDecision.CONTINUE)

    return EvaluationResult(
        decision=decision,
        reasoning=f"LLM evaluation failed ({error}), using fallback logic",
    )
