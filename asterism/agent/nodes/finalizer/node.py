"""Finalizer node implementation - generates final response to user."""

import logging

from asterism.agent.nodes.finalizer.response_builder import (
    build_error_response,
    build_success_response,
    format_results_summary,
)
from asterism.agent.nodes.shared import (
    LLMCaller,
    build_execution_trace,
    get_failed_tasks,
    get_user_request,
    set_final_response,
)
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def finalizer_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """Generate final response based on execution results.

    Args:
        llm: The LLM provider for synthesizing the response.
        state: Current agent state with completed execution.

    Returns:
        Updated state with final_response populated.
    """
    trace = build_execution_trace(state)
    failed_tasks = get_failed_tasks(state)

    if failed_tasks:
        logger.warning(f"[finalizer] Finalizing with {len(failed_tasks)} failed tasks")
        response = build_error_response(failed_tasks, trace)
        return set_final_response(state, response)

    return _build_success_finalization(state, trace, llm)


def _build_success_finalization(state: AgentState, trace: list[dict], llm: BaseLLMProvider) -> AgentState:
    """Build successful finalization with LLM-generated response."""
    logger.info(f"[finalizer] Generating success response for {len(trace)} tasks")

    caller = LLMCaller(llm, "finalizer_node")
    user_request = get_user_request(state)
    results_summary = format_results_summary(state)

    response, usage = build_success_response(state, trace, caller, user_request, results_summary)

    logger.info(f"[finalizer] Completed with {len(trace)} tasks, response length: {len(response.message)} chars")

    return set_final_response(state, response, usage)
