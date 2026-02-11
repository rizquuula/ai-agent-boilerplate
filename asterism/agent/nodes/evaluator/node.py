"""Evaluator node implementation - evaluates execution and routes next step."""

import logging

from asterism.agent.models import EvaluationDecision, EvaluationResult
from asterism.agent.nodes.evaluator.router import can_skip_evaluation, should_continue
from asterism.agent.nodes.evaluator.service import (
    apply_evaluation_result,
    create_fallback_evaluation,
    evaluate_with_llm,
)
from asterism.agent.nodes.shared import (
    set_evaluation_result,
)
from asterism.agent.state import AgentState
from asterism.llm.providers import BaseLLMProvider

logger = logging.getLogger(__name__)


def evaluator_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """Evaluate execution results and decide next action using LLM.

    For linear plans with all tasks completed successfully, this will skip
    the LLM evaluation and directly set the decision to finalize, saving
    tokens and reducing latency.

    Args:
        llm: The LLM provider for evaluation.
        state: Current agent state.

    Returns:
        Updated state with evaluation_result populated.
    """
    # Check if we can skip LLM evaluation for this state
    if can_skip_evaluation(state):
        logger.info("[evaluator] Skipping LLM evaluation for completed linear plan")

        # Create a finalize decision without LLM call
        evaluation = EvaluationResult(
            decision=EvaluationDecision.FINALIZE,
            reasoning="All tasks in linear plan completed successfully. Fast-path to finalization.",
            context_updates={},
            suggested_changes=None,
        )

        return set_evaluation_result(state, evaluation, None)

    logger.info("[evaluator] Starting evaluation")

    try:
        evaluation, usage = evaluate_with_llm(llm, state)
        return apply_evaluation_result(state, evaluation, usage, llm)

    except Exception as e:
        logger.error(f"[evaluator] LLM evaluation failed: {e}", exc_info=True)
        fallback = create_fallback_evaluation(state, str(e))
        return set_evaluation_result(state, fallback, None)


# Re-export for backward compatibility
__all__ = ["evaluator_node", "should_continue"]
