"""Evaluator node implementation."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from asterism.agent.models import EvaluationDecision, EvaluationResult, LLMUsage
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

from .prompts import EVALUATOR_SYSTEM_PROMPT
from .utils import build_evaluator_prompt, fallback_decision, resolve_next_task_inputs

logger = logging.getLogger(__name__)


def evaluator_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """
    Evaluate execution results and decide next action using LLM.

    The LLM will receive:
    1. SOUL.md + AGENT.md as a SystemMessage (loaded fresh from disk if configured)
    2. Node-specific evaluation instructions as a SystemMessage
    3. User request and execution context as a HumanMessage

    Args:
        llm: The LLM provider for evaluation.
        state: Current agent state.

    Returns:
        Updated state with evaluation_result populated.
    """
    # Build evaluation prompt
    user_prompt = build_evaluator_prompt(state)

    try:
        # Use structured output with message list
        # The provider will auto-prepend SOUL.md + AGENT.md as a SystemMessage
        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.debug(f"Evaluator sending prompt: {user_prompt[:500]}...")

        response = llm.invoke_structured(messages, EvaluationResult)
        evaluation = response.parsed

        logger.info(f"Evaluator decision: {evaluation.decision}, reasoning: {evaluation.reasoning[:200]}")

        # Update state with evaluation result
        new_state = state.copy()
        new_state["evaluation_result"] = evaluation

        # Track LLM usage from structured output response
        usage = LLMUsage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            model=llm.model,
            node_name="evaluator_node",
        )
        new_state["llm_usage"] = state.get("llm_usage", []) + [usage]

        # If replanning is needed, set error to trigger replanning
        if evaluation.decision == EvaluationDecision.REPLAN:
            # Build comprehensive error message for replanning
            error_parts = [f"Replanning needed: {evaluation.reasoning}"]

            # Include information about what failed
            execution_results = state.get("execution_results", [])
            if execution_results:
                last_result = execution_results[-1]
                if not last_result.success:
                    error_parts.append(f"Last task '{last_result.task_id}' failed: {last_result.error}")

            # Include suggested changes if available
            if evaluation.suggested_changes:
                error_parts.append(f"Suggested changes: {evaluation.suggested_changes}")

            new_state["error"] = "\n".join(error_parts)

            # Add detailed context to messages for planner
            replan_context = f"""[Evaluator] Replanning required.
Decision: {evaluation.decision}
Reasoning: {evaluation.reasoning}
Suggested changes: {evaluation.suggested_changes or "None provided"}"""
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=replan_context)]

        # If continuing, resolve next task inputs based on previous results
        if evaluation.decision == EvaluationDecision.CONTINUE:
            plan = new_state.get("plan")
            current_index = new_state.get("current_task_index", 0)
            if plan and current_index < len(plan.tasks):
                next_task = plan.tasks[current_index]
                # Only resolve if task has a tool call
                if next_task.tool_call:
                    logger.debug(f"Resolving inputs for task: {next_task.id}")
                    resolved_input, resolver_usage = resolve_next_task_inputs(llm, next_task, new_state)
                    if resolved_input is not None:
                        # Update the task's tool_input with resolved values
                        next_task.tool_input = resolved_input
                        logger.info(f"Resolved inputs for task {next_task.id}: {resolved_input}")
                    if resolver_usage:
                        # Track resolver LLM usage
                        new_state["llm_usage"] = new_state.get("llm_usage", []) + [resolver_usage]

        return new_state

    except Exception as e:
        logger.error(f"Evaluator LLM call failed: {e}")
        # Fallback to simple logic-based evaluation if LLM fails
        new_state = state.copy()
        fallback = fallback_decision(state)
        new_state["evaluation_result"] = EvaluationResult(
            decision=fallback,
            reasoning=f"LLM evaluation failed ({str(e)}), using fallback logic",
        )
        # If fallback suggests replanning, set error
        if fallback == EvaluationDecision.REPLAN:
            new_state["error"] = f"Evaluation failed, fallback to replan: {str(e)}"
        logger.info(f"Evaluator fallback decision: {fallback}")
        return new_state


def should_continue(state: AgentState) -> Literal["planner_node", "executor_node", "finalizer_node"]:
    """
    Routing function to determine next node after evaluation.

    Uses the evaluation_result from evaluator_node to decide routing.
    Falls back to simple logic if evaluation_result is not available.

    Args:
        state: Current agent state.

    Returns:
        Name of the next node to execute.
    """
    # Check for explicit error state first
    if state.get("error"):
        return "planner_node"

    # Use LLM evaluation result if available
    evaluation = state.get("evaluation_result")
    if evaluation:
        if evaluation.decision == EvaluationDecision.REPLAN:
            return "planner_node"
        elif evaluation.decision == EvaluationDecision.FINALIZE:
            return "finalizer_node"
        else:  # CONTINUE
            return "executor_node"

    # Fallback logic if no evaluation result
    plan = state.get("plan")
    if not plan:
        return "planner_node"

    current_index = state.get("current_task_index", 0)
    total_tasks = len(plan.tasks)

    if current_index >= total_tasks:
        return "finalizer_node"

    execution_results = state.get("execution_results", [])
    if execution_results and not execution_results[-1].success:
        return "planner_node"

    return "executor_node"
