"""Evaluator node implementation."""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from asterism.agent.models import EvaluationDecision, EvaluationResult, LLMUsage
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

from .prompts import EVALUATOR_SYSTEM_PROMPT
from .utils import build_evaluator_prompt, fallback_decision


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
        response = llm.invoke_structured(messages, EvaluationResult)
        evaluation = response.parsed

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
            new_state["error"] = f"Replanning needed: {evaluation.reasoning}"
            # Add suggested changes to messages for planner context
            if evaluation.suggested_changes:
                replan_context = f"Previous execution failed. Suggested changes: {evaluation.suggested_changes}"
                new_state["messages"] = state.get("messages", []) + [AIMessage(content=f"[Evaluator] {replan_context}")]

        return new_state

    except Exception as e:
        # Fallback to simple logic-based evaluation if LLM fails
        new_state = state.copy()
        new_state["evaluation_result"] = EvaluationResult(
            decision=fallback_decision(state),
            reasoning=f"LLM evaluation failed ({str(e)}), using fallback logic",
        )
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
