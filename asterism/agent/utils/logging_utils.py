"""Logging utilities for agent nodes with structured logging and timing."""

import functools
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from asterism.agent.state import AgentState

F = TypeVar("F", bound=Callable[..., AgentState])


def get_logger_context(state: AgentState, node_name: str) -> dict[str, Any]:
    """Build logging context from agent state.

    Args:
        state: Current agent state.
        node_name: Name of the current node.

    Returns:
        Dictionary with logging context.
    """
    context = {
        "node": node_name,
        "session_id": state.get("session_id", "unknown"),
        "trace_id": state.get("trace_id", "unknown"),
    }

    # Add plan context if available
    plan = state.get("plan")
    if plan:
        context["plan_task_count"] = len(plan.tasks)
        context["current_task_index"] = state.get("current_task_index", 0)

    # Add execution context
    execution_results = state.get("execution_results", [])
    if execution_results:
        context["completed_tasks"] = len(execution_results)
        last_result = execution_results[-1]
        context["last_task_id"] = last_result.task_id
        context["last_task_success"] = last_result.success

    # Add error context
    if state.get("error"):
        context["has_error"] = True
        context["error_preview"] = state["error"][:200] if state["error"] else None

    return context


def log_node_execution(node_name: str) -> Callable[[F], F]:
    """Decorator to log node entry, exit, and timing.

    Args:
        node_name: Name of the node for logging.

    Returns:
        Decorated function with logging.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> AgentState:
            # Extract state from args (llm, mcp_executor, state) or kwargs
            state: AgentState | None = None
            if len(args) >= 3:
                state = args[2]
            elif kwargs.get("state"):
                state = kwargs["state"]

            # Ensure trace_id exists
            if state and not state.get("trace_id"):
                state["trace_id"] = str(uuid.uuid4())

            # Build context
            context = get_logger_context(state, node_name) if state else {"node": node_name}

            # Log entry
            import logging

            logger = logging.getLogger(func.__module__)
            logger.info(
                f"[{node_name}] Node started",
                extra={
                    "agent_context": context,
                    "event": "node_started",
                },
            )

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Build result context
                result_context = {
                    **context,
                    "duration_ms": round(duration_ms, 2),
                    "event": "node_completed",
                }

                if result:
                    if result.get("error"):
                        result_context["error"] = result["error"][:500]
                    if result.get("evaluation_result"):
                        result_context["decision"] = result["evaluation_result"].decision

                logger.info(
                    f"[{node_name}] Node completed in {duration_ms:.2f}ms", extra={"agent_context": result_context}
                )

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                error_context = {
                    **context,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "event": "node_failed",
                }

                logger.error(
                    f"[{node_name}] Node failed: {e}",
                    extra={"agent_context": error_context},
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_llm_call(
    logger: logging.Logger,
    node_name: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    prompt_preview: str | None = None,
    response_preview: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log LLM call with structured information.

    Args:
        logger: Logger instance.
        node_name: Name of the node making the call.
        model: Model name used.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        duration_ms: Duration in milliseconds.
        prompt_preview: Preview of the prompt (first N chars).
        response_preview: Preview of the response (first N chars).
        success: Whether the call succeeded.
        error: Error message if failed.
    """
    context = {
        "node": node_name,
        "event": "llm_call",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration_ms": round(duration_ms, 2),
        "success": success,
    }

    if prompt_preview:
        context["prompt_preview"] = prompt_preview[:500]
    if response_preview:
        context["response_preview"] = response_preview[:500]
    if error:
        context["error"] = error

    if success:
        logger.info(
            f"[{node_name}] LLM call completed. Usage: ({prompt_tokens} + {completion_tokens} = {prompt_tokens + completion_tokens})",  # noqa: E501
            extra={"agent_context": context},
        )
    else:
        logger.error(f"[{node_name}] LLM call failed", extra={"agent_context": context})


def log_task_execution(
    logger: logging.Logger,
    task_id: str,
    task_type: str,
    success: bool,
    duration_ms: float,
    tool_call: str | None = None,
    error: str | None = None,
    result_preview: str | None = None,
) -> None:
    """Log task execution details.

    Args:
        logger: Logger instance.
        task_id: Task identifier.
        task_type: Type of task ("tool" or "llm").
        success: Whether execution succeeded.
        duration_ms: Execution duration in milliseconds.
        tool_call: Tool call string if applicable.
        error: Error message if failed.
        result_preview: Preview of the result.
    """
    context = {
        "event": "task_executed",
        "task_id": task_id,
        "task_type": task_type,
        "success": success,
        "duration_ms": round(duration_ms, 2),
    }

    if tool_call:
        context["tool_call"] = tool_call
    if error:
        context["error"] = error[:500]
    if result_preview:
        context["result_preview"] = result_preview[:500]

    if success:
        logger.info(f"[executor] Task {task_id} completed", extra={"agent_context": context})
    else:
        logger.warning(f"[executor] Task {task_id} failed", extra={"agent_context": context})


def log_plan_created(
    logger: logging.Logger,
    task_count: int,
    task_ids: list[str],
    has_dependencies: bool,
    reasoning_preview: str | None = None,
) -> None:
    """Log plan creation.

    Args:
        logger: Logger instance.
        task_count: Number of tasks in plan.
        task_ids: List of task IDs.
        has_dependencies: Whether any task has dependencies.
        reasoning_preview: Preview of the plan reasoning.
    """
    context = {
        "event": "plan_created",
        "task_count": task_count,
        "task_ids": task_ids,
        "has_dependencies": has_dependencies,
    }

    if reasoning_preview:
        context["reasoning_preview"] = reasoning_preview[:500]

    logger.info(f"[planner] Created plan with {task_count} tasks", extra={"agent_context": context})


def log_evaluation_decision(
    logger: logging.Logger,
    decision: str,
    reasoning_preview: str,
    suggested_changes: str | None = None,
) -> None:
    """Log evaluator decision.

    Args:
        logger: Logger instance.
        decision: The decision made (continue, replan, finalize).
        reasoning_preview: Preview of the reasoning.
        suggested_changes: Suggested changes if replanning.
    """
    context = {
        "event": "evaluation_decision",
        "decision": decision,
        "reasoning_preview": reasoning_preview[:500],
    }

    if suggested_changes:
        context["suggested_changes"] = suggested_changes[:500]

    logger.info(f"[evaluator] Decision: {decision}", extra={"agent_context": context})


def log_mcp_tool_call(
    logger: logging.Logger,
    server_name: str,
    tool_name: str,
    input_keys: list[str],
    success: bool,
    duration_ms: float,
    result_preview: str | None = None,
    error: str | None = None,
) -> None:
    """Log MCP tool call.

    Args:
        logger: Logger instance.
        server_name: MCP server name.
        tool_name: Tool name.
        input_keys: Keys of the input parameters.
        success: Whether the call succeeded.
        duration_ms: Duration in milliseconds.
        result_preview: Preview of the result.
        error: Error message if failed.
    """
    context = {
        "event": "mcp_tool_call",
        "server": server_name,
        "tool": tool_name,
        "input_keys": input_keys,
        "success": success,
        "duration_ms": round(duration_ms, 2),
    }

    if result_preview:
        context["result_preview"] = result_preview[:500]
    if error:
        context["error"] = error[:500]

    if success:
        logger.info(f"[executor] MCP tool {server_name}:{tool_name} succeeded", extra={"agent_context": context})
    else:
        logger.warning(f"[executor] MCP tool {server_name}:{tool_name} failed", extra={"agent_context": context})


def log_llm_call_start(
    logger: logging.Logger,
    node_name: str,
    model: str,
    action: str,
    prompt_preview: str | None = None,
) -> None:
    """Log the start of an LLM call.

    This is useful for tracking which node is making an LLM call and what
    action it's performing, especially useful for debugging slow/hanging calls.

    Args:
        logger: Logger instance.
        node_name: Name of the node making the call.
        model: Model name being used.
        action: Description of what the LLM call is for (e.g., "planning", "evaluating").
        prompt_preview: Preview of the prompt (first N chars).
    """
    context = {
        "node": node_name,
        "event": "llm_call_start",
        "model": model,
        "action": action,
    }

    if prompt_preview:
        context["prompt_preview"] = prompt_preview[:500]

    logger.info(f"[{node_name}] Starting LLM call for {action}", extra={"agent_context": context})
