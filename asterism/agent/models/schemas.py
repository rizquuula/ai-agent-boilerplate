"""Pydantic models for agent state and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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


class TaskResult(BaseModel):
    """Result of executing a single task."""

    task_id: str = Field(..., description="ID of the completed task")
    success: bool = Field(..., description="Whether the task succeeded")
    result: Any = Field(default=None, description="Result data from the task")
    error: str | None = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the task completed")


class AgentResponse(BaseModel):
    """Final structured response from the agent."""

    message: str = Field(..., description="Natural language response to the user")
    execution_trace: list[dict[str, Any]] = Field(
        ..., description="Full execution history with task details and results"
    )
    plan_used: Plan | None = Field(default=None, description="The plan that was executed")
