"""Finalizer node implementation."""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import AgentResponse, LLMUsage
from asterism.agent.state import AgentState
from asterism.agent.utils import log_llm_call
from asterism.llm.base import BaseLLMProvider

from .prompts import FINALIZER_SYSTEM_PROMPT
from .utils import get_user_request

logger = logging.getLogger(__name__)


def finalizer_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """
    Generate final response based on execution results.

    The LLM will receive:
    1. SOUL.md + AGENT.md as a SystemMessage (loaded fresh from disk if configured)
    2. Node-specific response synthesis instructions as a SystemMessage
    3. User request and execution results as a HumanMessage

    Args:
        llm: The LLM provider for synthesizing the response.
        state: Current agent state with completed execution.

    Returns:
        Updated state with final_response populated.
    """
    plan = state.get("plan")
    execution_results = state.get("execution_results", [])

    logger.info(f"[finalizer] Starting finalization with {len(execution_results)} execution results")

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
    usage = None

    if failed_tasks:
        logger.warning(f"[finalizer] Finalizing with {len(failed_tasks)} failed tasks")
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

            user_prompt = f"""Original user request: {get_user_request(state)}

Execution results:
{results_summary}

Create a response for the user."""

            # Use message list - the provider will auto-prepend SOUL.md + AGENT.md
            messages = [
                SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            logger.debug(f"[finalizer] Sending LLM request with prompt preview: {user_prompt[:300]}...")

            # Time the LLM call
            start_time = time.perf_counter()
            llm_response = llm.invoke_with_usage(messages)
            duration_ms = (time.perf_counter() - start_time) * 1000

            response = AgentResponse(
                message=llm_response.content,
                execution_trace=execution_trace,
                plan_used=plan,
            )

            # Track LLM usage
            usage = LLMUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                model=llm.model,
                node_name="finalizer_node",
            )

            # Log LLM call
            log_llm_call(
                logger=logger,
                node_name="finalizer_node",
                model=llm.model,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                duration_ms=duration_ms,
                prompt_preview=user_prompt[:500],
                response_preview=llm_response.content[:500],
                success=True,
            )

            logger.info(f"[finalizer] Generated response with {len(execution_results)} execution trace entries")
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000 if "start_time" in locals() else 0

            # Log failed LLM call
            log_llm_call(
                logger=logger,
                node_name="finalizer_node",
                model=llm.model,
                prompt_tokens=getattr(llm_response, "prompt_tokens", 0) if "llm_response" in locals() else 0,
                completion_tokens=getattr(llm_response, "completion_tokens", 0) if "llm_response" in locals() else 0,
                duration_ms=duration_ms,
                prompt_preview=user_prompt[:500] if "user_prompt" in locals() else None,
                success=False,
                error=str(e),
            )

            # Fallback if LLM fails
            logger.error(f"[finalizer] LLM call failed: {e}", exc_info=True)
            response = AgentResponse(
                message=f"Task completed successfully, but response generation failed: {str(e)}",
                execution_trace=execution_trace,
                plan_used=plan,
            )
            usage = None

    new_state = state.copy()
    new_state["final_response"] = response
    new_state["error"] = None

    # Add LLM usage to state if available
    if usage:
        new_state["llm_usage"] = state.get("llm_usage", []) + [usage]

    # Log finalization summary
    logger.info(
        f"[finalizer] Completed with {len(execution_results)} tasks, "
        f"{len(failed_tasks)} failed, response length: {len(response.message)} chars"
    )
    return new_state
