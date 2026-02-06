"""Evaluator node for deciding next steps after task execution."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import EvaluationDecision, EvaluationResult, LLMUsage
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

# Node-specific system prompt - this is combined with SOUL.md + AGENT.md
EVALUATOR_SYSTEM_PROMPT = """You are an execution evaluator for an AI agent. 
Your job is to analyze task execution results and decide the next action.

You will receive:
- The user's original request
- The current plan with remaining tasks
- Execution history with results and errors
- Current task context

Evaluate and decide:
1. **continue** - Proceed to next task (execution on track)
2. **replan** - Current plan needs adjustment (unexpected results, failures, new information)
3. **finalize** - Goals achieved, can complete early (all critical tasks done, user satisfied)

Guidelines:
- Be conservative with replanning - only if execution significantly deviated
- Consider partial successes - some results may be good enough
- Check if user goal is satisfied even if not all tasks completed
- Provide clear reasoning for your decision

Return JSON with decision and reasoning."""


def _get_user_request(state: AgentState) -> str:
    """Extract the original user request from state messages."""
    from langchain_core.messages import HumanMessage

    messages = state.get("messages", [])
    if messages:
        # Find the first human message
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
    return "No user request found"


def _format_tasks(tasks: list, current_index: int) -> str:
    """Format tasks list with completion status."""
    lines = []
    for i, task in enumerate(tasks):
        status = "[✓]" if i < current_index else "[ ]"
        state_label = "COMPLETED" if i < current_index else ("NEXT" if i == current_index else "PENDING")
        tool_info = f" (tool: {task.tool_call})" if task.tool_call else " (LLM task)"
        lines.append(f"{status} {task.id}: {task.description}{tool_info} ({state_label})")
    return "\n".join(lines)


def _format_execution_results(results: list) -> str:
    """Format execution results for the prompt."""
    if not results:
        return "No tasks executed yet."

    lines = []
    for result in results:
        status = "✓" if result.success else "✗"
        if result.success:
            # Truncate long results
            result_str = str(result.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "... [truncated]"
            lines.append(f"{status} {result.task_id}: {result_str}")
        else:
            lines.append(f"{status} {result.task_id}: ERROR - {result.error}")
    return "\n".join(lines)


def _build_evaluator_prompt(state: AgentState) -> str:
    """Build the evaluation prompt from state."""
    plan = state.get("plan")
    execution_results = state.get("execution_results", [])
    current_index = state.get("current_task_index", 0)

    user_request = _get_user_request(state)

    if not plan:
        return f"""=== USER REQUEST ===
{user_request}

=== CURRENT STATE ===
No plan exists. Replanning required.

=== DECISION ===
Decision: replan
Reasoning: No plan available to execute."""

    # Get last result info
    last_result = execution_results[-1] if execution_results else None

    return f"""=== USER REQUEST ===
{user_request}

=== CURRENT PLAN ===
Total tasks: {len(plan.tasks)}
Completed: {current_index}/{len(plan.tasks)}
Remaining: {len(plan.tasks) - current_index}

Plan reasoning: {plan.reasoning}

Tasks:
{_format_tasks(plan.tasks, current_index)}

=== EXECUTION HISTORY ===
{_format_execution_results(execution_results)}

=== CURRENT CONTEXT ===
Last task: {last_result.task_id if last_result else "N/A"}
Last result: {str(last_result.result)[:200] if last_result and last_result.result else "N/A"}
Last error: {last_result.error if last_result and last_result.error else "None"}

=== DECISION REQUIRED ===
Based on the execution so far, should we:
1. **continue** - Proceed to next task (execution on track)
2. **replan** - Current plan needs adjustment (unexpected results, failures, new information)
3. **finalize** - Goals achieved, can complete early (all critical tasks done, user satisfied)

Provide your evaluation with clear reasoning."""


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
    user_prompt = _build_evaluator_prompt(state)

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
                from langchain_core.messages import AIMessage

                replan_context = f"Previous execution failed. Suggested changes: {evaluation.suggested_changes}"
                new_state["messages"] = state.get("messages", []) + [AIMessage(content=f"[Evaluator] {replan_context}")]

        return new_state

    except Exception as e:
        # Fallback to simple logic-based evaluation if LLM fails
        new_state = state.copy()
        new_state["evaluation_result"] = EvaluationResult(
            decision=_fallback_decision(state),
            reasoning=f"LLM evaluation failed ({str(e)}), using fallback logic",
        )
        return new_state


def _fallback_decision(state: AgentState) -> EvaluationDecision:
    """Fallback decision logic when LLM evaluation fails."""
    plan = state.get("plan")
    if not plan:
        return EvaluationDecision.REPLAN

    current_index = state.get("current_task_index", 0)
    if current_index >= len(plan.tasks):
        return EvaluationDecision.FINALIZE

    execution_results = state.get("execution_results", [])
    if execution_results and not execution_results[-1].success:
        return EvaluationDecision.REPLAN

    return EvaluationDecision.CONTINUE


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
