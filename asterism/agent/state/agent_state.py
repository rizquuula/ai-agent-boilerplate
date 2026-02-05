"""Agent state definition using TypedDict."""

from typing import TypedDict

from langchain_core.messages import BaseMessage

from asterism.agent.models import AgentResponse, Plan, TaskResult


class AgentState(TypedDict):
    """State for the agent workflow."""

    session_id: str
    messages: list[BaseMessage]
    plan: Plan | None
    current_task_index: int
    execution_results: list[TaskResult]
    final_response: AgentResponse | None
    error: str | None
