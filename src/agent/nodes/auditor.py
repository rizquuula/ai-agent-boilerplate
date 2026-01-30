"""Auditor node: Success validation and replanning using LLM."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent.state import AgentState, ErrorRecord, SubGoal
from agent.utils.tool_parser import (
    format_execution_results,
    format_milestone_history,
    get_milestone_history,
    has_execution_failures,
)

if TYPE_CHECKING:
    from agent.llm.base import BaseLLMProvider


class ValidationResult(BaseModel):
    """Result of milestone validation."""

    success: bool = Field(description="Whether the milestone was completed successfully")
    reasoning: str = Field(description="Explanation of the validation decision")
    retry_strategy: str | None = Field(default=None, description="Suggested approach if retry needed")


def create_validation_prompt(state: AgentState) -> str:
    """Create the validation prompt for the LLM.

    Args:
        state: Current agent state

    Returns:
        Formatted prompt string
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    if not milestones or current_idx >= len(milestones):
        return "No active milestone to validate."

    current_milestone = milestones[current_idx]
    history = state.get("history", [])
    milestone_history = get_milestone_history(history, current_milestone.id)

    prompt_parts = [
        "You are a validation AI. Determine if the milestone was completed successfully.",
        "",
        "MILESTONE TO VALIDATE:",
        f"Description: {current_milestone.description}",
        f"Success Criteria: {current_milestone.success_criteria}",
        "",
    ]

    # Include execution history
    if milestone_history:
        prompt_parts.extend(
            [
                "TOOLS EXECUTED:",
                format_milestone_history(milestone_history),
                "",
                "EXECUTION RESULTS:",
                format_execution_results(milestone_history),
                "",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "TOOLS EXECUTED: None",
                "",
            ]
        )

    prompt_parts.extend(
        [
            "VALIDATION INSTRUCTIONS:",
            "1. Check if the success criteria were met",
            "2. Verify tools executed successfully (no errors)",
            "3. Confirm the milestone's objective was achieved",
            "4. Consider partial successes appropriately",
            "",
            "RESPONSE FORMAT:",
            "- success: true/false (strict - only true if criteria fully met)",
            "- reasoning: Detailed explanation of your decision",
            "- retry_strategy: Specific approach to fix issues (if failed)",
            "",
            "MAKE YOUR VALIDATION:",
        ]
    )

    return "\n".join(prompt_parts)


def handle_retry(state: AgentState, milestone: SubGoal, validation: ValidationResult | None = None) -> AgentState:
    """Handle retry logic with limits.

    Args:
        state: Current agent state
        milestone: Current milestone
        validation: Validation result if available

    Returns:
        Updated state with retry information
    """
    retry_count = state.get("retry_count", 0) + 1
    max_retries = 3

    errors = state.get("errors", [])

    if retry_count > max_retries:
        # Max retries exceeded - mark as failed and move on
        milestone.status = "failed"

        # Add error record for max retries
        error_record = ErrorRecord(
            milestone_id=milestone.id,
            error_message=f"Max retries ({max_retries}) exceeded for milestone",
            suggested_fix="Manual intervention required",
        )
        errors.append(error_record)

        return {
            **state,
            "last_verification_status": "failed",
            "retry_count": 0,
            "errors": errors,
        }

    # Add error record with retry strategy
    if validation and validation.retry_strategy:
        error_record = ErrorRecord(
            milestone_id=milestone.id,
            error_message=validation.reasoning,
            suggested_fix=validation.retry_strategy,
        )
        errors.append(error_record)

    return {
        **state,
        "last_verification_status": "failed",
        "retry_count": retry_count,
        "errors": errors,
    }


def update_global_context(current_context: str, milestone: SubGoal, history: list) -> str:
    """Update global context with milestone results.

    Args:
        current_context: Existing global context
        milestone: Completed milestone
        history: Execution history for this milestone

    Returns:
        Updated global context string
    """
    # Count successes and failures
    successful_tools = sum(1 for h in history if h.success)
    failed_tools = sum(1 for h in history if not h.success)

    milestone_summary = f"""
Milestone: {milestone.description}
Status: {"COMPLETED" if milestone.status == "completed" else "FAILED"}
Tools: {successful_tools} successful, {failed_tools} failed
"""

    # Combine with existing context
    if current_context:
        return f"{current_context}\n{milestone_summary}"
    return milestone_summary


def node(state: AgentState, llm_provider: "BaseLLMProvider") -> AgentState:
    """Validate milestone completion using LLM.

    Args:
        state: Current agent state with execution history
        llm_provider: LLM provider for validation

    Returns:
        Updated state with validation status
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    # Validate we have an active milestone
    if not milestones or current_idx >= len(milestones):
        return {**state, "last_verification_status": "failed"}

    current_milestone = milestones[current_idx]
    history = state.get("history", [])
    milestone_history = get_milestone_history(history, current_milestone.id)

    # Check for execution failures first
    if has_execution_failures(milestone_history):
        # Even before LLM validation, we know something failed
        # Still run LLM validation to get retry strategy
        pass

    try:
        # Generate validation prompt
        prompt = create_validation_prompt(state)

        # Use LLM to validate
        validation = llm_provider.invoke_structured(prompt, ValidationResult)

        if validation.success:
            # Mark milestone as completed
            current_milestone.status = "completed"

            # Update global context with learnings
            global_context = state.get("global_context", "")
            updated_global = update_global_context(global_context, current_milestone, milestone_history)

            return {
                **state,
                "last_verification_status": "passed",
                "retry_count": 0,
                "global_context": updated_global,
            }
        else:
            # Handle retry
            return handle_retry(state, current_milestone, validation)

    except Exception as e:
        # Handle LLM errors gracefully - fail open and retry
        error_record = ErrorRecord(
            milestone_id=current_milestone.id,
            error_message=f"Validation LLM call failed: {str(e)}",
            suggested_fix="Retry with same approach",
        )

        errors = state.get("errors", [])
        errors.append(error_record)

        return handle_retry({**state, "errors": errors}, current_milestone, None)
