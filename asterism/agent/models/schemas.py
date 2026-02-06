"""Pydantic models for agent state and responses."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationDecision(StrEnum):
    """Decision options for the evaluator."""

    CONTINUE = "continue"
    REPLAN = "replan"
    FINALIZE = "finalize"


class Task(BaseModel):
    """A single task in a plan."""

    id: str = Field(..., description="Unique identifier for this task")
    description: str = Field(..., description="Human-readable description of the task")
    tool_call: str | None = Field(
        default=None,
        description="Tool call in format 'server_name:tool_name', or None for LLM-only tasks",
    )
    tool_input: dict[str, Any] | None = Field(default=None, description="Input parameters for the tool call")
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task",
    )


class Plan(BaseModel):
    """A plan consisting of multiple tasks."""

    tasks: list[Task] = Field(..., description="Ordered list of tasks to execute")
    reasoning: str = Field(..., description="Explanation of the plan's approach")


class LLMUsage(BaseModel):
    """LLM token usage for a single call."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")
    model: str = Field(..., description="Model name used for the call")
    node_name: str = Field(..., description="Node that made the call (planner, executor, evaluator, finalizer)")


class UsageSummary(BaseModel):
    """Summary of total LLM usage across all calls."""

    total_prompt_tokens: int = Field(default=0, description="Total prompt tokens across all calls")
    total_completion_tokens: int = Field(default=0, description="Total completion tokens across all calls")
    total_tokens: int = Field(default=0, description="Total tokens across all calls")
    calls_by_node: dict[str, int] = Field(default_factory=dict, description="Number of calls per node type")


class TaskResult(BaseModel):
    """Result of executing a single task."""

    task_id: str = Field(..., description="ID of the completed task")
    success: bool = Field(..., description="Whether the task succeeded")
    result: Any = Field(default=None, description="Result data from the task")
    error: str | None = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the task completed")
    llm_usage: LLMUsage | None = Field(default=None, description="LLM usage if task used LLM")


class EvaluationResult(BaseModel):
    """Result of evaluating execution progress."""

    decision: EvaluationDecision = Field(..., description="Decision: continue, replan, or finalize")
    reasoning: str = Field(..., description="Explanation of why this decision was made")
    context_updates: dict[str, Any] = Field(
        default_factory=dict, description="Optional context updates to pass to next node"
    )
    suggested_changes: str | None = Field(default=None, description="If replanning, suggestions for what to change")


class AgentResponse(BaseModel):
    """Final structured response from the agent."""

    message: str = Field(..., description="Natural language response to the user")
    execution_trace: list[dict[str, Any]] = Field(
        ..., description="Full execution history with task details and results"
    )
    plan_used: Plan | None = Field(default=None, description="The plan that was executed")
    total_usage: UsageSummary = Field(
        default_factory=UsageSummary, description="Total LLM token usage across all nodes"
    )
