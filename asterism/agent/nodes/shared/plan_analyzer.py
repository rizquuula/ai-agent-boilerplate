"""Plan structure analyzer for optimization opportunities."""

from asterism.agent.models import Plan


def is_linear_plan(plan: Plan | None) -> bool:
    """Check if a plan is a simple linear/sequential plan.

    A linear plan has tasks where each task (after the first) depends only
    on the immediately preceding task, forming a simple chain:
    task_1 → task_2 → task_3 → ...

    Args:
        plan: The plan to analyze.

    Returns:
        True if the plan is linear/sequential, False otherwise.
    """
    if not plan or not plan.tasks:
        return False

    tasks = plan.tasks

    # Single task is linear
    if len(tasks) == 1:
        return True

    # Build a map of task IDs to their index
    # task_indices = {task.id: i for i, task in enumerate(tasks)}

    for i, task in enumerate(tasks):
        if i == 0:
            # First task should have no dependencies
            if task.depends_on:
                return False
        else:
            # Each subsequent task should depend ONLY on the previous task
            expected_dep = tasks[i - 1].id
            if task.depends_on != [expected_dep]:
                return False

    return True


def get_execution_batch(plan: Plan, current_index: int) -> list[int]:
    """Get the range of task indices that can be executed in a batch.

    For linear plans, this returns all remaining tasks.
    For non-linear plans, returns only the current task.

    Args:
        plan: The execution plan.
        current_index: The current task index.

    Returns:
        List of task indices that can be executed together.
    """
    if not plan or not plan.tasks:
        return []

    if not is_linear_plan(plan):
        # Non-linear plans execute one at a time
        if current_index < len(plan.tasks):
            return [current_index]
        return []

    # Linear plan - can execute all remaining tasks
    return list(range(current_index, len(plan.tasks)))


def can_skip_intermediate_evaluation(plan: Plan, current_index: int, all_succeeded: bool) -> bool:
    """Determine if intermediate evaluation can be skipped.

    For linear plans with all tasks succeeding so far, we can skip
    intermediate evaluations and go directly to the next task.

    Args:
        plan: The execution plan.
        current_index: The index of the next task to execute.
        all_succeeded: Whether all executed tasks have succeeded.

    Returns:
        True if evaluation can be skipped, False otherwise.
    """
    if not plan or not plan.tasks:
        return False

    # Only skip if plan is linear and all tasks succeeded
    if not is_linear_plan(plan) or not all_succeeded:
        return False

    # Don't skip if we're at the last task (need final evaluation)
    if current_index >= len(plan.tasks):
        return False

    return True


def should_finalize_directly(plan: Plan, current_index: int, all_succeeded: bool) -> bool:
    """Check if we can go directly to finalizer without evaluation.

    When all tasks in a linear plan are complete and successful,
    we can skip the final evaluator call.

    Args:
        plan: The execution plan.
        current_index: The current task index.
        all_succeeded: Whether all executed tasks have succeeded.

    Returns:
        True if we can finalize directly, False otherwise.
    """
    if not plan or not plan.tasks:
        return False

    if not is_linear_plan(plan) or not all_succeeded:
        return False

    # All tasks completed
    if current_index >= len(plan.tasks):
        return True

    return False


def analyze_plan_complexity(plan: Plan | None) -> dict:
    """Analyze plan and return complexity metrics.

    Args:
        plan: The plan to analyze.

    Returns:
        Dictionary with complexity analysis.
    """
    if not plan or not plan.tasks:
        return {
            "is_linear": False,
            "task_count": 0,
            "max_parallel": 0,
            "can_batch_execute": False,
        }

    tasks = plan.tasks
    is_linear = is_linear_plan(plan)

    # Calculate max parallel tasks (tasks with same dependency level)
    dependency_depths = {}
    for task in tasks:
        if not task.depends_on:
            dependency_depths[task.id] = 0
        else:
            max_dep_depth = max(dependency_depths.get(dep, 0) for dep in task.depends_on)
            dependency_depths[task.id] = max_dep_depth + 1

    depth_counts = {}
    for depth in dependency_depths.values():
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    max_parallel = max(depth_counts.values()) if depth_counts else 1

    return {
        "is_linear": is_linear,
        "task_count": len(tasks),
        "max_parallel": max_parallel,
        "can_batch_execute": is_linear and len(tasks) > 1,
        "dependency_levels": len(depth_counts),
    }
