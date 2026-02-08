"""Response building logic for the finalizer node."""

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import AgentResponse, LLMUsage
from asterism.agent.nodes.finalizer.prompts import FINALIZER_SYSTEM_PROMPT
from asterism.agent.nodes.shared import LLMCaller
from asterism.agent.state import AgentState


def build_error_response(failed_tasks: list, trace: list[dict]) -> AgentResponse:
    """Build response when tasks have failed.

    Args:
        failed_tasks: List of failed task results.
        trace: Execution trace.

    Returns:
        AgentResponse with error message.
    """
    error_messages = [f"Task {r.task_id} failed: {r.error}" for r in failed_tasks]

    message = f"The task could not be completed due to {len(failed_tasks)} error(s).\n\n" + "\n".join(error_messages)

    return AgentResponse(
        message=message,
        execution_trace=trace,
        plan_used=None,
    )


def build_success_response(
    state: AgentState,
    trace: list[dict],
    caller: LLMCaller,
    user_request: str,
    results_summary: str,
) -> tuple[AgentResponse, LLMUsage | None]:
    """Build response using LLM for successful execution.

    Args:
        state: Current agent state.
        trace: Execution trace.
        caller: LLM caller instance.
        user_request: The original user request.
        results_summary: Summary of execution results.

    Returns:
        Tuple of (AgentResponse, LLMUsage or None if LLM call failed).
    """
    user_prompt = f"""Original user request: {user_request}

Execution results:
{results_summary}

Create a response for the user."""

    messages = [
        SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        result = caller.call_text(messages, "generating final response")

        response = AgentResponse(
            message=result.parsed,
            execution_trace=trace,
            plan_used=state.get("plan"),
        )

        return response, result.usage

    except Exception as e:
        # Fallback if LLM fails
        fallback_message = f"Task completed successfully, but response generation failed: {str(e)}"

        response = AgentResponse(
            message=fallback_message,
            execution_trace=trace,
            plan_used=state.get("plan"),
        )

        return response, None


def format_results_summary(state: AgentState) -> str:
    """Format execution results as summary for LLM.

    Args:
        state: Current agent state.

    Returns:
        Formatted summary string.
    """
    execution_results = state.get("execution_results", [])
    if not execution_results:
        return "No execution results."

    return "\n".join(f"Task {r.task_id}: {r.result}" for r in execution_results)
