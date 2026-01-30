"""Architect node: High-level milestone planner using LLM."""

import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent.state import AgentState, ErrorRecord, SubGoal
from agent.utils.skill_loader import get_available_skills_description

if TYPE_CHECKING:
    from agent.llm.base import BaseLLMProvider


class MilestoneItem(BaseModel):
    """Single milestone item from LLM planning."""

    description: str = Field(description="What needs to be accomplished")
    skill: str = Field(description="Skill to use for this milestone")
    success_criteria: str = Field(description="How to verify completion")


class MilestonePlan(BaseModel):
    """Complete milestone plan from LLM."""

    milestones: list[MilestoneItem] = Field(description="List of milestones to complete")
    reasoning: str = Field(description="Explanation of the planning approach")


def create_planning_prompt(objective: str) -> str:
    """Create the planning prompt for the LLM.

    Args:
        objective: The user's objective

    Returns:
        Formatted prompt string
    """
    return f"""You are a strategic planning AI.
Your task is to break down a user objective into a sequence of high-level milestones.

OBJECTIVE:
{objective}

AVAILABLE SKILLS:
{get_available_skills_description()}

INSTRUCTIONS:
1. Break down the objective into 3-7 logical milestones
2. Each milestone should be concrete and achievable
3. Assign the most appropriate skill to each milestone from the available skills
4. Define clear success criteria for each milestone
5. Milestones should be sequential - each builds on previous ones

GUIDELINES:
- Start with information gathering/discovery if needed
- Group related operations together
- End with synthesis or reporting if appropriate
- Be specific about what each milestone produces

RESPONSE FORMAT:
Return a structured plan with:
- description: Clear statement of what the milestone accomplishes
- skill: One of the available skills (filesystem, ppt-writer, song-writer, etc.)
- success_criteria: Specific, verifiable criteria for completion

Example good milestone:
- description: "Find all Python files in the project directory"
- skill: "filesystem"
- success_criteria: "A list of .py file paths is obtained, even if empty"

PLAN THE MILESTONES NOW:"""


def node(state: AgentState, llm_provider: "BaseLLMProvider") -> AgentState:
    """Generate high-level milestones using LLM planning.

    Args:
        state: Current agent state with objective
        llm_provider: LLM provider for planning

    Returns:
        Updated state with generated milestones
    """
    objective = state.get("objective", "")

    if not objective:
        # Handle empty objective gracefully
        return {
            **state,
            "milestones": [],
            "current_idx": 0,
            "tactical_plan": [],
            "execution_context": {},
            "retry_count": 0,
            "errors": [],
            "global_context": "",
        }

    try:
        # Generate planning prompt
        prompt = create_planning_prompt(objective)

        # Use LLM to generate structured plan
        plan: MilestonePlan = llm_provider.invoke_structured(prompt, MilestonePlan)

        # Convert to SubGoal objects
        milestones = [
            SubGoal(
                id=str(uuid.uuid4()),
                description=m.description,
                assigned_skill=m.skill,
                status="pending",
                success_criteria=m.success_criteria,
            )
            for m in plan.milestones
        ]

        return {
            **state,
            "milestones": milestones,
            "current_idx": 0,
            "tactical_plan": [],
            "execution_context": {},
            "retry_count": 0,
            "errors": [],
            "global_context": f"Planning reasoning: {plan.reasoning}",
        }

    except Exception as e:
        # Handle LLM errors gracefully with fallback milestones
        error_record = ErrorRecord(
            milestone_id="planning",
            error_message=f"LLM planning failed: {str(e)}",
            suggested_fix="Check LLM provider configuration and retry",
        )

        return {
            **state,
            "milestones": [],
            "current_idx": 0,
            "tactical_plan": [],
            "execution_context": {},
            "retry_count": 0,
            "errors": [error_record],
            "global_context": "",
        }
