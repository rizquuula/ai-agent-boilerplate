"""Finalizer node for generating the final response."""

from asterism.agent.models import AgentResponse
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider


def finalizer_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """
    Generate final response based on execution results.

    Args:
        llm: The LLM provider for synthesizing the response.
        state: Current agent state with completed execution.

    Returns:
        Updated state with final_response populated.
    """
    plan = state.get("plan")
    execution_results = state.get("execution_results", [])

    # Build execution trace
    execution_trace = []
    for result in execution_results:
        trace_entry = {
            "task_id": result.task_id,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
        }
        execution_trace.append(trace_entry)

    # Check if there were any errors
    failed_tasks = [r for r in execution_results if not r.success]

    if failed_tasks:
        # Generate error response
        error_messages = [f"Task {r.task_id} failed: {r.error}" for r in failed_tasks]
        response = AgentResponse(
            message=f"The task could not be completed due to {len(failed_tasks)} error(s).\n\n"
            + "\n".join(error_messages),
            execution_trace=execution_trace,
            plan_used=plan,
        )
    else:
        # Generate success response using LLM
        try:
            # Summarize results for LLM
            results_summary = "\n".join(f"Task {r.task_id}: {r.result}" for r in execution_results)

            system_prompt = """You are a helpful assistant that synthesizes task execution results
into a clear, concise response for the user.

Provide a natural language answer that:
- Directly addresses the user's original request
- Summarizes what was accomplished
- Highlights key findings or outcomes
- Is friendly and professional

Do not include technical details like task IDs or execution traces in the message - those are provided separately."""

            user_prompt = f"""Original user request: {_get_user_request(state)}

Execution results:
{results_summary}

Create a response for the user."""

            message = llm.invoke(user_prompt, system_message=system_prompt)

            response = AgentResponse(
                message=message,
                execution_trace=execution_trace,
                plan_used=plan,
            )
        except Exception as e:
            # Fallback if LLM fails
            response = AgentResponse(
                message=f"Task completed successfully, but response generation failed: {str(e)}",
                execution_trace=execution_trace,
                plan_used=plan,
            )

    new_state = state.copy()
    new_state["final_response"] = response
    new_state["error"] = None

    return new_state


def _get_user_request(state: AgentState) -> str:
    """Extract the original user request from state messages."""
    messages = state.get("messages", [])
    if messages:
        # Find the first human message
        from langchain_core.messages import HumanMessage

        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
    return "No user request found"
