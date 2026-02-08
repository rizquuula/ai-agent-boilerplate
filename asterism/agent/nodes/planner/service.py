"""Planning business logic and validation."""

import logging

from asterism.agent.models import Plan
from asterism.agent.utils import log_plan_created

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Error raised when planning fails."""

    pass


def validate_and_enrich_plan(plan: Plan) -> Plan:
    """Validate plan and ensure all tasks have IDs.

    Args:
        plan: The plan from LLM.

    Returns:
        Validated and enriched plan.

    Raises:
        PlanningError: If plan is invalid.
    """
    if not plan:
        raise PlanningError("Plan is empty")

    if not plan.tasks:
        raise PlanningError("Plan has no tasks")

    # Ensure all tasks have IDs
    for i, task in enumerate(plan.tasks):
        if not task.id:
            task.id = _generate_task_id(i, task.description)

    return plan


def log_plan_creation(plan: Plan) -> None:
    """Log plan creation with structured context.

    Args:
        plan: The created plan.
    """
    has_dependencies = any(t.depends_on for t in plan.tasks)

    log_plan_created(
        logger=logger,
        task_count=len(plan.tasks),
        task_ids=[t.id for t in plan.tasks],
        has_dependencies=has_dependencies,
        reasoning_preview=plan.reasoning,
    )


def _generate_task_id(index: int, description: str) -> str:
    """Generate a unique task ID.

    Args:
        index: Task index in plan.
        description: Task description.

    Returns:
        Generated task ID.
    """
    safe_desc = description.lower().replace(" ", "_")[:30]
    return f"task_{index}_{safe_desc}"
