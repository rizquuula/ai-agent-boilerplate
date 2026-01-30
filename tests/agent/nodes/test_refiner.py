"""Unit tests for the refiner node."""

from agent.nodes import refiner
from agent.nodes.refiner import TacticalPlan, TacticalStep
from agent.state import ErrorRecord, ToolCall


class TestRefinerNode:
    """Tests for the refiner node."""

    def test_refiner_generates_tactical_plan(self, mock_llm_provider, sample_agent_state):
        """Should break milestone into ToolCall list."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Filesystem skill context"

        mock_plan = TacticalPlan(
            steps=[
                TacticalStep(
                    tool="filesystem:list_files",
                    parameters={"path": ".", "pattern": "*.py"},
                    expected_outcome="List Python files",
                ),
            ],
            reasoning="Need to find Python files",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = refiner.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert len(result["tactical_plan"]) == 1
        assert isinstance(result["tactical_plan"][0], ToolCall)
        assert result["tactical_plan"][0].tool_id == "filesystem:list_files"
        assert result["tactical_plan"][0].parameters == {"path": ".", "pattern": "*.py"}

    def test_refiner_considers_history_on_retry(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should adapt plan based on previous errors."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Filesystem context"
        sample_agent_state["history"] = [sample_execution_record]
        sample_agent_state["retry_count"] = 1
        sample_agent_state["errors"] = [
            ErrorRecord(
                milestone_id=sample_agent_state["milestones"][0].id,
                error_message="Permission denied",
                suggested_fix="Try different directory",
            )
        ]

        mock_plan = TacticalPlan(
            steps=[TacticalStep(tool="filesystem:list_files", parameters={}, expected_outcome="Retry")],
            reasoning="Retry with different approach",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        _ = refiner.node(sample_agent_state, mock_llm_provider)

        # Assert
        mock_llm_provider.invoke_structured.assert_called_once()
        # Verify the prompt included error context
        call_args = mock_llm_provider.invoke_structured.call_args
        prompt = call_args[0][0]
        assert "RETRY ATTEMPT" in prompt or "Previous attempts failed" in prompt

    def test_refiner_includes_tool_parameters(self, mock_llm_provider, sample_agent_state):
        """ToolCall should include parameters dict."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Context"

        mock_plan = TacticalPlan(
            steps=[
                TacticalStep(
                    tool="filesystem:read_file",
                    parameters={"path": "test.py", "encoding": "utf-8"},
                    expected_outcome="Read file",
                ),
            ],
            reasoning="Read a file",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = refiner.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["tactical_plan"][0].parameters == {
            "path": "test.py",
            "encoding": "utf-8",
        }

    def test_refiner_limits_plan_length(self, mock_llm_provider, sample_agent_state):
        """Should generate reasonable number of steps."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Context"

        # Generate many steps
        mock_plan = TacticalPlan(
            steps=[TacticalStep(tool=f"tool{i}", parameters={}, expected_outcome=f"Step {i}") for i in range(10)],
            reasoning="Many steps",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        result = refiner.node(sample_agent_state, mock_llm_provider)

        # Assert - should accept what LLM returns (even if many steps)
        assert len(result["tactical_plan"]) == 10

    def test_refiner_handles_no_active_milestone(self, mock_llm_provider):
        """Should handle when no milestone is active."""
        # Arrange
        state = {"milestones": [], "current_idx": 0}

        # Act
        result = refiner.node(state, mock_llm_provider)

        # Assert
        assert result["tactical_plan"] == []
        mock_llm_provider.invoke_structured.assert_not_called()

    def test_refiner_handles_llm_error(self, mock_llm_provider, sample_agent_state):
        """Should handle LLM errors gracefully."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Context"
        mock_llm_provider.invoke_structured.side_effect = Exception("LLM error")

        # Act
        result = refiner.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["tactical_plan"] == []
        assert len(result["errors"]) == 1
        assert "Tactical planning failed" in result["errors"][0].error_message

    def test_refiner_includes_execution_context(self, mock_llm_provider, sample_agent_state):
        """Should include execution context in prompt."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Context"
        sample_agent_state["execution_context"] = {"discovered_files": ["main.py"]}

        mock_plan = TacticalPlan(
            steps=[TacticalStep(tool="tool", parameters={}, expected_outcome="result")],
            reasoning="Test",
        )
        mock_llm_provider.invoke_structured.return_value = mock_plan

        # Act
        refiner.node(sample_agent_state, mock_llm_provider)

        # Assert
        call_args = mock_llm_provider.invoke_structured.call_args
        prompt = call_args[0][0]
        assert "EXECUTION CONTEXT" in prompt or "discovered_files" in prompt


class TestCreateRefinerPrompt:
    """Tests for the create_refiner_prompt function."""

    def test_prompt_includes_milestone_details(self, sample_agent_state):
        """Prompt should include milestone description and criteria."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Skill context"

        # Act
        prompt = refiner.create_refiner_prompt(sample_agent_state)

        # Assert
        assert "List all Python files" in prompt
        assert "Success Criteria:" in prompt

    def test_prompt_includes_skill_context(self, sample_agent_state):
        """Prompt should include the active skill context."""
        # Arrange
        sample_agent_state["active_skill_context"] = "Filesystem skill guidance"

        # Act
        prompt = refiner.create_refiner_prompt(sample_agent_state)

        # Assert
        assert "Filesystem skill guidance" in prompt

    def test_prompt_no_milestone(self):
        """Should handle missing milestone gracefully."""
        # Arrange
        state = {"milestones": [], "current_idx": 0}

        # Act
        prompt = refiner.create_refiner_prompt(state)

        # Assert
        assert "No active milestone" in prompt
