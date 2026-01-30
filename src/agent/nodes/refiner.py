"""Refiner node: Tactical planner that breaks milestones into tool calls using LLM."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent.state import AgentState, ErrorRecord, ToolCall
from agent.utils.tool_parser import (
    format_errors,
    format_execution_context,
    format_milestone_history,
    get_milestone_history,
)

if TYPE_CHECKING:
    from agent.llm.base import BaseLLMProvider


class TacticalStep(BaseModel):
    """Single step in the tactical plan."""

    tool: str = Field(description="Tool identifier in format 'server:tool'")
    parameters: dict = Field(default_factory=dict, description="Tool parameters")
    expected_outcome: str = Field(description="What this step should achieve")


class TacticalPlan(BaseModel):
    """Complete tactical plan from LLM."""

    steps: list[TacticalStep] = Field(description="Ordered list of tool calls")
    reasoning: str = Field(description="Explanation of the approach")


def create_refiner_prompt(state: AgentState) -> str:
    """Create the tactical planning prompt for the LLM.

    Args:
        state: Current agent state

    Returns:
        Formatted prompt string
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    if not milestones or current_idx >= len(milestones):
        return "No active milestone to plan for."

    current_milestone = milestones[current_idx]
    history = state.get("history", [])
    milestone_history = get_milestone_history(history, current_milestone.id)
    retry_count = state.get("retry_count", 0)

    prompt_parts = [
        state.get("active_skill_context", ""),
        "",
        "TACTICAL PLANNING TASK:",
        "Break down the milestone into 1-5 specific tool calls.",
        "",
        f"Milestone: {current_milestone.description}",
        f"Success Criteria: {current_milestone.success_criteria}",
        "",
    ]

    # Include execution history if any
    if milestone_history:
        prompt_parts.extend(
            [
                "TOOLS ALREADY EXECUTED FOR THIS MILESTONE:",
                format_milestone_history(milestone_history),
                "",
            ]
        )

    # Include error context if retrying
    if retry_count > 0:
        errors = state.get("errors", [])
        prompt_parts.extend(
            [
                f"RETRY ATTEMPT #{retry_count}:",
                format_errors([e.model_dump() for e in errors[-3:]]),
                "",
                "Adjust your approach based on previous errors.",
                "",
            ]
        )

    # Include execution context
    execution_context = state.get("execution_context", {})
    if execution_context:
        prompt_parts.extend(
            [
                "EXECUTION CONTEXT (data from previous milestones):",
                format_execution_context(execution_context),
                "",
            ]
        )

    prompt_parts.extend(
        [
            "TOOL CALL FORMAT:",
            '- tool: "server:tool_name" (e.g., "filesystem:list_files")',
            "- parameters: JSON object with tool arguments",
            "- expected_outcome: What this tool call should produce",
            "",
            "GUIDELINES:",
            "1. Use available tools from the skill context above",
            "2. Each step should be atomic and verifiable",
            "3. Pass parameters as proper JSON, not strings",
            '4. For file patterns, use proper glob syntax (e.g., "**/*.py")',
            "5. If retrying, try a different approach based on errors",
            "6. Maximum 5 steps - combine operations when possible",
            "",
            "GENERATE THE TACTICAL PLAN:",
        ]
    )

    return "\n".join(prompt_parts)


def node(state: AgentState, llm_provider: "BaseLLMProvider") -> AgentState:
    """Generate tactical plan using LLM.

    Args:
        state: Current agent state with active milestone
        llm_provider: LLM provider for planning

    Returns:
        Updated state with tactical_plan set
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    # Validate we have an active milestone
    if not milestones or current_idx >= len(milestones):
        return {**state, "tactical_plan": []}

    try:
        # Generate planning prompt
        prompt = create_refiner_prompt(state)

        # Use LLM to generate structured plan
        plan = llm_provider.invoke_structured(prompt, TacticalPlan)

        # Convert to ToolCall objects
        tactical_plan = [
            ToolCall(
                tool_id=step.tool,
                parameters=step.parameters,
                expected_outcome=step.expected_outcome,
            )
            for step in plan.steps
        ]

        return {**state, "tactical_plan": tactical_plan}

    except Exception as e:
        # Handle LLM errors gracefully
        current_milestone = milestones[current_idx]
        error_record = ErrorRecord(
            milestone_id=current_milestone.id,
            error_message=f"Tactical planning failed: {str(e)}",
            suggested_fix="Retry with simplified approach",
        )

        errors = state.get("errors", [])
        errors.append(error_record)

        return {
            **state,
            "tactical_plan": [],
            "errors": errors,
        }
