"""Shared components for agent nodes.

This module provides cross-cutting concerns used across all nodes:
- LLM invocation with standardized logging
- State manipulation utilities
- Context extraction from agent state
- Execution trace building
"""

from .context_extractors import (
    are_dependencies_satisfied,
    format_execution_history,
    get_current_task,
    get_failed_tasks,
    get_last_result,
    get_user_request,
    has_execution_history,
)
from .llm_caller import LLMCaller, LLMCallResult
from .state_utils import (
    advance_task,
    append_llm_usage,
    create_error_state,
    prepare_replan_state,
    set_evaluation_result,
    set_final_response,
    set_plan,
)
from .trace_builder import build_execution_trace

__all__ = [
    # LLM Caller
    "LLMCaller",
    "LLMCallResult",
    # Context Extractors
    "get_user_request",
    "get_last_result",
    "format_execution_history",
    "has_execution_history",
    "get_current_task",
    "get_failed_tasks",
    "are_dependencies_satisfied",
    # State Utils
    "create_error_state",
    "append_llm_usage",
    "advance_task",
    "prepare_replan_state",
    "set_evaluation_result",
    "set_final_response",
    "set_plan",
    # Trace Builder
    "build_execution_trace",
]
