"""Agent state definition using TypedDict."""

from typing import TypedDict

from langchain_core.messages import BaseMessage

from asterism.agent.models import AgentResponse, EvaluationResult, LLMUsage, Plan, TaskResult


class AgentState(TypedDict):
    """State for the agent workflow."""

    session_id: str
    trace_id: str | None  # Unique trace ID for correlating logs across the entire flow
    messages: list[BaseMessage]
    plan: Plan | None
    current_task_index: int
    execution_results: list[TaskResult]
    evaluation_result: EvaluationResult | None
    final_response: AgentResponse | None
    error: str | None
    llm_usage: list[LLMUsage]
