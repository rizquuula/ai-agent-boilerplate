"""Unit tests for plan analyzer utilities."""

from asterism.agent.models import Plan, Task
from asterism.agent.nodes.shared.plan_analyzer import (
    analyze_plan_complexity,
    can_skip_intermediate_evaluation,
    get_execution_batch,
    is_linear_plan,
    should_finalize_directly,
)


class TestIsLinearPlan:
    """Test cases for is_linear_plan function."""

    def test_empty_plan_is_not_linear(self):
        """Empty plan should not be considered linear."""
        assert is_linear_plan(None) is False
        assert is_linear_plan(Plan(tasks=[], reasoning="empty")) is False

    def test_single_task_is_linear(self):
        """Single task with no dependencies is linear."""
        plan = Plan(tasks=[Task(id="task_1", description="Do something")], reasoning="simple task")
        assert is_linear_plan(plan) is True

    def test_linear_chain_is_linear(self):
        """Simple chain A → B → C is linear."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
                Task(id="task_3", description="Step 3", depends_on=["task_2"]),
            ],
            reasoning="linear chain",
        )
        assert is_linear_plan(plan) is True

    def test_branching_is_not_linear(self):
        """Plan with branching (two tasks depending on same task) is not linear."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2a", description="Step 2a", depends_on=["task_1"]),
                Task(id="task_2b", description="Step 2b", depends_on=["task_1"]),
            ],
            reasoning="branched",
        )
        assert is_linear_plan(plan) is False

    def test_multiple_dependencies_is_not_linear(self):
        """Task with multiple dependencies is not linear."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2", description="Step 2", depends_on=[]),
                Task(id="task_3", description="Step 3", depends_on=["task_1", "task_2"]),
            ],
            reasoning="merge",
        )
        assert is_linear_plan(plan) is False

    def test_first_task_with_dependencies_is_not_linear(self):
        """First task should not have dependencies for linear plan."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=["nonexistent"]),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="bad start",
        )
        assert is_linear_plan(plan) is False

    def test_skipped_dependency_is_not_linear(self):
        """Task depending on non-immediate predecessor is not linear."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2", description="Step 2", depends_on=[]),
                Task(id="task_3", description="Step 3", depends_on=["task_1"]),  # Skips task_2
            ],
            reasoning="skipped",
        )
        assert is_linear_plan(plan) is False


class TestGetExecutionBatch:
    """Test cases for get_execution_batch function."""

    def test_empty_plan_returns_empty(self):
        """Empty plan returns empty batch."""
        assert get_execution_batch(Plan(tasks=[], reasoning="empty"), 0) == []
        assert get_execution_batch(None, 0) == []

    def test_non_linear_plan_returns_single_task(self):
        """Non-linear plans execute one task at a time."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2a", description="Step 2a", depends_on=["task_1"]),
                Task(id="task_2b", description="Step 2b", depends_on=["task_1"]),
            ],
            reasoning="branched",
        )
        assert get_execution_batch(plan, 0) == [0]
        assert get_execution_batch(plan, 1) == [1]

    def test_linear_plan_returns_all_remaining(self):
        """Linear plans can batch all remaining tasks."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1", depends_on=[]),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
                Task(id="task_3", description="Step 3", depends_on=["task_2"]),
            ],
            reasoning="linear",
        )
        # From start, all tasks can be batched
        assert get_execution_batch(plan, 0) == [0, 1, 2]
        # From middle, remaining tasks can be batched
        assert get_execution_batch(plan, 1) == [1, 2]
        # Last task
        assert get_execution_batch(plan, 2) == [2]

    def test_out_of_bounds_returns_empty(self):
        """Index beyond task count returns empty batch."""
        plan = Plan(tasks=[Task(id="task_1", description="Step 1")], reasoning="single")
        assert get_execution_batch(plan, 5) == []


class TestCanSkipIntermediateEvaluation:
    """Test cases for can_skip_intermediate_evaluation function."""

    def test_empty_plan_cannot_skip(self):
        """Empty plan cannot skip evaluation."""
        assert can_skip_intermediate_evaluation(None, 0, True) is False

    def test_non_linear_plan_cannot_skip(self):
        """Non-linear plan cannot skip evaluation."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2a", description="Step 2a", depends_on=["task_1"]),
                Task(id="task_2b", description="Step 2b", depends_on=["task_1"]),
            ],
            reasoning="branched",
        )
        assert can_skip_intermediate_evaluation(plan, 1, True) is False

    def test_failed_task_cannot_skip(self):
        """If any task failed, cannot skip evaluation."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="linear",
        )
        assert can_skip_intermediate_evaluation(plan, 1, False) is False

    def test_all_tasks_completed_cannot_skip(self):
        """When all tasks are done, we need final evaluation, not intermediate skip."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="linear",
        )
        assert can_skip_intermediate_evaluation(plan, 2, True) is False

    def test_linear_plan_with_pending_tasks_can_skip(self):
        """Linear plan with more tasks to do can skip intermediate evaluation."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
                Task(id="task_3", description="Step 3", depends_on=["task_2"]),
            ],
            reasoning="linear",
        )
        # After first task, can skip to execute second
        assert can_skip_intermediate_evaluation(plan, 1, True) is True
        # After second task, can skip to execute third
        assert can_skip_intermediate_evaluation(plan, 2, True) is True


class TestShouldFinalizeDirectly:
    """Test cases for should_finalize_directly function."""

    def test_empty_plan_cannot_finalize(self):
        """Empty plan cannot finalize directly."""
        assert should_finalize_directly(None, 0, True) is False

    def test_non_linear_plan_cannot_finalize(self):
        """Non-linear plan cannot finalize directly."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2a", description="Step 2a", depends_on=["task_1"]),
                Task(id="task_2b", description="Step 2b", depends_on=["task_1"]),
            ],
            reasoning="branched",
        )
        assert should_finalize_directly(plan, 3, True) is False

    def test_failed_task_cannot_finalize(self):
        """If tasks failed, cannot finalize directly."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="linear",
        )
        assert should_finalize_directly(plan, 2, False) is False

    def test_incomplete_tasks_cannot_finalize(self):
        """If tasks remain, cannot finalize directly."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="linear",
        )
        assert should_finalize_directly(plan, 1, True) is False

    def test_completed_linear_plan_can_finalize(self):
        """Completed linear plan can finalize directly."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
            ],
            reasoning="linear",
        )
        assert should_finalize_directly(plan, 2, True) is True


class TestAnalyzePlanComplexity:
    """Test cases for analyze_plan_complexity function."""

    def test_empty_plan(self):
        """Empty plan analysis."""
        result = analyze_plan_complexity(None)
        assert result["is_linear"] is False
        assert result["task_count"] == 0
        assert result["can_batch_execute"] is False

    def test_single_task(self):
        """Single task plan analysis."""
        plan = Plan(tasks=[Task(id="task_1", description="Step 1")], reasoning="simple")
        result = analyze_plan_complexity(plan)
        assert result["is_linear"] is True
        assert result["task_count"] == 1
        assert result["can_batch_execute"] is False  # Single task, no benefit
        assert result["dependency_levels"] == 1

    def test_linear_plan(self):
        """Linear plan analysis."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2", description="Step 2", depends_on=["task_1"]),
                Task(id="task_3", description="Step 3", depends_on=["task_2"]),
            ],
            reasoning="linear",
        )
        result = analyze_plan_complexity(plan)
        assert result["is_linear"] is True
        assert result["task_count"] == 3
        assert result["can_batch_execute"] is True
        assert result["max_parallel"] == 1
        assert result["dependency_levels"] == 3

    def test_parallel_plan(self):
        """Plan with parallel tasks analysis."""
        plan = Plan(
            tasks=[
                Task(id="task_1", description="Step 1"),
                Task(id="task_2a", description="Step 2a", depends_on=["task_1"]),
                Task(id="task_2b", description="Step 2b", depends_on=["task_1"]),
            ],
            reasoning="parallel",
        )
        result = analyze_plan_complexity(plan)
        assert result["is_linear"] is False
        assert result["task_count"] == 3
        assert result["can_batch_execute"] is False
        assert result["max_parallel"] == 2
        assert result["dependency_levels"] == 2
