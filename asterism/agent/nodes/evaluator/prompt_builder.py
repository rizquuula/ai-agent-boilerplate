"""Prompt building for evaluator node."""

from asterism.agent.state import AgentState


def build_evaluator_prompt(state: AgentState) -> str:
    """Build evaluation prompt from state.

    Args:
        state: Current agent state.

    Returns:
        Formatted prompt string for evaluator LLM.
    """
    user_request = _extract_user_request(state)
    plan_info = _build_plan_info(state)
    execution_history = _build_execution_history(state)
    current_context = _build_current_context(state)

    return f"""=== USER REQUEST ===
{user_request}

=== CURRENT PLAN ===
{plan_info}

=== EXECUTION HISTORY ===
{execution_history}

=== CURRENT CONTEXT ===
{current_context}

=== DECISION REQUIRED ===
Based on the execution so far, should we:
1. **continue** - Proceed to next task (execution on track)
2. **replan** - Current plan needs adjustment (unexpected results, failures, new information)
3. **finalize** - Goals achieved, can complete early (all critical tasks done, user satisfied)

Provide your evaluation with clear reasoning."""


def _extract_user_request(state: AgentState) -> str:
    """Extract user request from state messages."""
    from asterism.agent.nodes.shared import get_user_request

    return get_user_request(state)


def _build_plan_info(state: AgentState) -> str:
    """Build plan status information."""
    plan = state.get("plan")
    if not plan:
        return "No plan exists. Replanning required.\n\nDecision: replan\nReasoning: No plan available to execute."

    current_index = state.get("current_task_index", 0)
    total = len(plan.tasks)

    lines = [
        f"Total tasks: {total}",
        f"Completed: {current_index}/{total}",
        f"Remaining: {total - current_index}",
        f"\nPlan reasoning: {plan.reasoning}",
        "\nTasks:",
    ]

    for i, task in enumerate(plan.tasks):
        status = "[✓]" if i < current_index else "[ ]"
        state_label = "COMPLETED" if i < current_index else ("NEXT" if i == current_index else "PENDING")
        tool_info = f" (tool: {task.tool_call})" if task.tool_call else " (LLM task)"
        lines.append(f"{status} {task.id}: {task.description}{tool_info} ({state_label})")

    return "\n".join(lines)


def _build_execution_history(state: AgentState) -> str:
    """Build execution results history."""
    results = state.get("execution_results", [])
    if not results:
        return "No tasks executed yet."

    lines = []
    for result in results:
        status = "✓" if result.success else "✗"
        if result.success:
            result_str = str(result.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "... [truncated]"
            lines.append(f"{status} {result.task_id}: {result_str}")
        else:
            lines.append(f"{status} {result.task_id}: ERROR - {result.error}")

    return "\n".join(lines)


def _build_current_context(state: AgentState) -> str:
    """Build current task context."""
    results = state.get("execution_results", [])
    if not results:
        return "Last task: N/A\nLast result: N/A\nLast error: None"

    last = results[-1]
    result_preview = str(last.result)[:200] if last.result else "N/A"

    return f"""Last task: {last.task_id}
Last result: {result_preview}
Last error: {last.error or "None"}"""
