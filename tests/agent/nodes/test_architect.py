"""Unit tests for the architect node."""

from agent.nodes import architect
from agent.nodes.architect import MilestoneItem, MilestonePlan
from agent.state import SubGoal


class TestArchitectNode:
    """Tests for the architect node."""

    def test_architect_generates_milestones_from_objective(self, mock_llm_provider, sample_agent_state):
        """Given an objective, should generate list of SubGoals."""
        # Arrange
        mock_plan = MilestonePlan(
            milestones=[
                MilestoneItem(
                    description="Find all Python files",
                    skill="filesystem",
                    success_criteria="Python files are listed",
                ),
                MilestoneItem(
                    description="Analyze code complexity",
                    skill="code-analyzer",
                    success_criteria="Complexity metrics calculated",
                ),
            ],
            reasoning="Standard code analysis workflow",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = architect.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert len(result["milestones"]) == 2
        assert all(isinstance(m, SubGoal) for m in result["milestones"])
        assert result["milestones"][0].description == "Find all Python files"
        assert result["milestones"][0].assigned_skill == "filesystem"
        assert result["current_idx"] == 0
        mock_llm_provider.invoke_structured.assert_called_once()

    def test_architect_handles_empty_objective(self, mock_llm_provider):
        """Should handle empty/invalid objectives gracefully."""
        # Arrange
        state = {
            "objective": "",
            "milestones": [],
            "current_idx": 0,
            "tactical_plan": [],
            "history": [],
            "active_skill_context": "",
            "last_verification_status": "",
            "execution_context": {},
            "retry_count": 0,
            "global_context": "",
            "errors": [],
        }

        # Act
        result = architect.node(state, mock_llm_provider)

        # Assert
        assert result["milestones"] == []
        assert result["current_idx"] == 0
        mock_llm_provider.invoke_structured.assert_not_called()

    def test_architect_includes_success_criteria(self, mock_llm_provider, sample_agent_state):
        """Generated milestones should include success criteria."""
        # Arrange
        mock_plan = MilestonePlan(
            milestones=[
                MilestoneItem(
                    description="Test milestone",
                    skill="filesystem",
                    success_criteria="Specific success condition",
                ),
            ],
            reasoning="Test reasoning",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = architect.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["milestones"][0].success_criteria == "Specific success condition"

    def test_architect_resets_state_fields(self, mock_llm_provider, sample_agent_state):
        """Should reset execution state for new run."""
        # Arrange
        sample_agent_state["execution_context"] = {"old": "data"}
        sample_agent_state["retry_count"] = 2
        sample_agent_state["errors"] = [{"error": "old error"}]
        sample_agent_state["tactical_plan"] = ["old plan"]

        mock_plan = MilestonePlan(
            milestones=[MilestoneItem(description="Test", skill="filesystem", success_criteria="Done")],
            reasoning="Test",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = architect.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["current_idx"] == 0
        assert result["execution_context"] == {}
        assert result["retry_count"] == 0
        assert result["errors"] == []
        assert result["tactical_plan"] == []

    def test_architect_handles_llm_error(self, mock_llm_provider, sample_agent_state):
        """Should handle LLM errors gracefully."""
        # Arrange
        mock_llm_provider.invoke_structured.side_effect = Exception("LLM API error")

        # Act
        result = architect.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["milestones"] == []
        assert len(result["errors"]) == 1
        assert "LLM planning failed" in result["errors"][0].error_message

    def test_architect_generates_unique_ids(self, mock_llm_provider, sample_agent_state):
        """Each milestone should have a unique ID."""
        # Arrange
        mock_plan = MilestonePlan(
            milestones=[
                MilestoneItem(description="Task 1", skill="filesystem", success_criteria="Done"),
                MilestoneItem(description="Task 2", skill="filesystem", success_criteria="Done"),
            ],
            reasoning="Test",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = architect.node(sample_agent_state, mock_llm_provider)

        # Assert
        ids = [m.id for m in result["milestones"]]
        assert len(ids) == len(set(ids))  # All IDs are unique


class TestCreatePlanningPrompt:
    """Tests for the create_planning_prompt function."""

    def test_prompt_includes_objective(self):
        """Prompt should include the objective."""
        # Act
        prompt = architect.create_planning_prompt("Test objective")

        # Assert
        assert "Test objective" in prompt
        assert "OBJECTIVE:" in prompt

    def test_prompt_includes_available_skills(self):
        """Prompt should mention available skills."""
        # Act
        prompt = architect.create_planning_prompt("Test objective")

        # Assert
        assert "AVAILABLE SKILLS:" in prompt

    def test_prompt_has_instructions(self):
        """Prompt should include instructions for the LLM."""
        # Act
        prompt = architect.create_planning_prompt("Test objective")

        # Assert
        assert "INSTRUCTIONS:" in prompt
        assert "break down the objective" in prompt.lower()
