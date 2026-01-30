"""Router node: Skill and MCP tool selector."""

from agent.state import AgentState
from agent.utils.skill_loader import (
    format_completed_milestones,
    load_skill_config,
    normalize_skill_name,
)
from agent.utils.tool_parser import format_available_tools, format_execution_context


def node(state: AgentState) -> AgentState:
    """Load skill configuration and set up execution context.

    Args:
        state: Current agent state with milestones

    Returns:
        Updated state with active_skill_context set
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    # Check if we have valid milestones
    if not milestones or current_idx >= len(milestones):
        return {
            **state,
            "active_skill_context": "No active milestone. Waiting for planning.",
        }

    current_milestone = milestones[current_idx]
    skill_name = current_milestone.assigned_skill

    try:
        # Load skill configuration (normalizes skill name internally)
        skill_config = load_skill_config(skill_name)

        # Build comprehensive skill context
        skill_context = f"""You are the {skill_config.name} specialist.

{skill_config.description}

CURRENT MILESTONE:
Description: {current_milestone.description}
Success Criteria: {current_milestone.success_criteria}

EXECUTION CONTEXT:
{format_execution_context(state.get("execution_context", {}))}

COMPLETED MILESTONES:
{format_completed_milestones(milestones, current_idx)}

AVAILABLE TOOLS:
{format_available_tools()}

INSTRUCTIONS:
- Focus on completing the current milestone using available tools
- Follow the success criteria when determining completion
- Use execution context to build upon previous work
- If tools fail, document what went wrong for retry

SKILL GUIDANCE:
{skill_config.content[:500]}...
"""

        return {
            **state,
            "active_skill_context": skill_context.strip(),
            "retry_count": 0,  # Reset retry count for new milestone
        }

    except FileNotFoundError:
        # Handle missing skill configuration gracefully
        fallback_context = f"""You are operating with the "{skill_name}" skill.

CURRENT MILESTONE:
Description: {current_milestone.description}
Success Criteria: {current_milestone.success_criteria}

COMPLETED MILESTONES:
{format_completed_milestones(milestones, current_idx)}

AVAILABLE TOOLS:
{format_available_tools()}

Note: Skill configuration files not found for '{normalize_skill_name(skill_name)}'. 
Using basic capabilities. You should still attempt to complete the milestone.
"""

        return {
            **state,
            "active_skill_context": fallback_context.strip(),
            "retry_count": 0,
        }

    except Exception as e:
        # Handle any other errors
        error_context = f"""You are operating with the "{skill_name}" skill.

CURRENT MILESTONE: {current_milestone.description}

ERROR LOADING SKILL: {str(e)}

Please proceed with basic capabilities to complete the milestone.
"""

        return {
            **state,
            "active_skill_context": error_context.strip(),
            "retry_count": 0,
        }
