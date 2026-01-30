"""State definitions for the hierarchical agent workflow."""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SubGoal(BaseModel):
    """Represents a high-level milestone in the task execution."""

    id: str = Field(description="Unique identifier for the milestone")
    description: str = Field(description="What needs to be accomplished")
    assigned_skill: str = Field(description="Skill to use for this milestone")
    status: Literal["pending", "active", "completed", "failed"] = Field(
        default="pending", description="Current status of the milestone"
    )
    success_criteria: str = Field(default="", description="Criteria to determine if milestone is complete")


class ToolCall(BaseModel):
    """Represents a single tool invocation."""

    tool_id: str = Field(description="Tool identifier in format 'server:tool'")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool call")
    expected_outcome: str = Field(default="", description="What should happen when this tool runs")


class ExecutionRecord(BaseModel):
    """Record of a tool execution."""

    milestone_id: str = Field(description="ID of the milestone this execution belongs to")
    tool_call: ToolCall = Field(description="The tool call that was executed")
    result: dict[str, Any] = Field(description="Execution result from MCP")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = Field(description="Whether the execution succeeded")


class ErrorRecord(BaseModel):
    """Record of an execution error."""

    milestone_id: str = Field(description="ID of the milestone where error occurred")
    error_message: str = Field(description="Description of what went wrong")
    suggested_fix: str = Field(default="", description="Suggested strategy to fix")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(TypedDict):
    """Global state for the hierarchical agent workflow."""

    # Core fields
    objective: str  # The original user request
    milestones: list[SubGoal]  # The high-level plan
    current_idx: int  # Index of the active milestone
    tactical_plan: list[ToolCall]  # Atomic steps for the current skill
    history: list[ExecutionRecord]  # Tool results & observations
    active_skill_context: str  # Current system prompt injection
    last_verification_status: str  # "passed" or "failed" for validation

    # Enhanced fields for production readiness
    execution_context: dict[str, Any]  # Accumulated knowledge across milestones
    retry_count: int  # Track retries per milestone
    global_context: str  # Summary of all completed work
    errors: list[ErrorRecord]  # Structured error tracking


# Type aliases for clarity
Milestones = list[SubGoal]
TacticalPlan = list[ToolCall]
History = list[ExecutionRecord]
Errors = list[ErrorRecord]
