"""Utility functions for the evaluator node."""

from langchain_core.messages import HumanMessage

from asterism.agent.models import EvaluationDecision
from asterism.agent.state import AgentState


def get_user_request(state: AgentState) -> str:
    """Extract the original user request from state messages."""
    messages = state.get("messages", [])
    if messages:
        # Find the first human message
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
    return "No user request found"


def format_tasks(tasks: list, current_index: int) -> str:
    """Format tasks list with completion status."""
    lines = []
    for i, task in enumerate(tasks):
        status = "[✓]" if i < current_index else "[ ]"
        state_label = "COMPLETED" if i < current_index else ("NEXT" if i == current_index else "PENDING")
        tool_info = f" (tool: {task.tool_call})" if task.tool_call else " (LLM task)"
        lines.append(f"{status} {task.id}: {task.description}{tool_info} ({state_label})")
    return "\n".join(lines)


def format_execution_results(results: list) -> str:
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


def build_evaluator_prompt(state: AgentState) -> str:
    """Build the evaluation prompt from state."""
    plan = state.get("plan")
    execution_results = state.get("execution_results", [])
    current_index = state.get("current_task_index", 0)

    user_request = get_user_request(state)

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
{format_tasks(plan.tasks, current_index)}

=== EXECUTION HISTORY ===
{format_execution_results(execution_results)}

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


def fallback_decision(state: AgentState) -> EvaluationDecision:
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
