"""Agent state definition using TypedDict."""

from typing import TypedDict

from langchain_core.messages import BaseMessage

from asterism.agent.models import AgentResponse, EvaluationResult, Plan, TaskResult


class AgentState(TypedDict):
    """State for the agent workflow."""

    session_id: str
    messages: list[BaseMessage]
    plan: Plan | None
    current_task_index: int
    execution_results: list[TaskResult]
    evaluation_result: EvaluationResult | None
    final_response: AgentResponse | None
    error: str | None
