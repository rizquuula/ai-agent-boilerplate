"""Unit tests for the auditor node."""

from agent.nodes import auditor
from agent.nodes.auditor import ValidationResult
from agent.state import ErrorRecord, ExecutionRecord, SubGoal, ToolCall


class TestAuditorNode:
    """Tests for the auditor node."""

    def test_auditor_validates_success(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should mark passed when milestone complete."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]

        mock_validation = ValidationResult(
            success=True,
            reasoning="All tools executed successfully",
            retry_strategy=None,
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["last_verification_status"] == "passed"
        assert result["milestones"][0].status == "completed"
        assert result["retry_count"] == 0

    def test_auditor_detects_failure(self, mock_llm_provider, sample_agent_state):
        """Should mark failed when tools failed."""
        # Arrange
        sample_agent_state["history"] = [
            ExecutionRecord(
                milestone_id=sample_agent_state["milestones"][0].id,
                tool_call=ToolCall(tool_id="tool", parameters={}, expected_outcome=""),
                result={"success": False, "error": "Failed"},
                success=False,
            )
        ]

        mock_validation = ValidationResult(
            success=False,
            reasoning="Tool execution failed",
            retry_strategy="Check permissions and retry",
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["last_verification_status"] == "failed"
        assert result["retry_count"] == 1

    def test_auditor_handles_retry(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should increment retry_count on failure."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]
        sample_agent_state["retry_count"] = 1  # Already retried once

        mock_validation = ValidationResult(
            success=False,
            reasoning="Still failed",
            retry_strategy="Try different approach",
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["retry_count"] == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0].suggested_fix == "Try different approach"

    def test_auditor_max_retries_exceeded(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should mark failed and move on after max retries."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]
        sample_agent_state["retry_count"] = 3  # Max retries already

        mock_validation = ValidationResult(
            success=False,
            reasoning="Still failed after retries",
            retry_strategy=None,
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["last_verification_status"] == "failed"
        assert result["retry_count"] == 0  # Reset for next milestone
        assert result["milestones"][0].status == "failed"
        assert len(result["errors"]) == 1  # Max retries error added

    def test_auditor_updates_global_context(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should update global context on success."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]

        mock_validation = ValidationResult(
            success=True,
            reasoning="All good",
            retry_strategy=None,
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert "Milestone:" in result["global_context"]
        assert "COMPLETED" in result["global_context"]

    def test_auditor_handles_llm_error(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should handle LLM errors gracefully."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]
        mock_llm_provider.invoke_structured.side_effect = Exception("LLM error")

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert result["last_verification_status"] == "failed"
        assert result["retry_count"] == 1
        assert len(result["errors"]) == 1
        assert "Validation LLM call failed" in result["errors"][0].error_message

    def test_auditor_no_active_milestone(self, mock_llm_provider):
        """Should handle state with no milestones."""
        # Arrange
        state = {"milestones": [], "current_idx": 0, "history": [], "errors": []}

        # Act
        result = auditor.node(state, mock_llm_provider)

        # Assert
        assert result["last_verification_status"] == "failed"
        mock_llm_provider.invoke_structured.assert_not_called()

    def test_auditor_preserves_error_history(self, mock_llm_provider, sample_agent_state, sample_execution_record):
        """Should preserve existing errors when adding new ones."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]
        sample_agent_state["errors"] = [
            ErrorRecord(
                milestone_id=sample_agent_state["milestones"][0].id,
                error_message="Previous error",
                suggested_fix="Previous fix",
            )
        ]

        mock_validation = ValidationResult(
            success=False,
            reasoning="New failure",
            retry_strategy="New fix",
        )
        mock_llm_provider.invoke_structured.return_value = mock_validation

        # Act
        result = auditor.node(sample_agent_state, mock_llm_provider)

        # Assert
        assert len(result["errors"]) == 2
        assert result["errors"][0].error_message == "Previous error"


class TestHandleRetry:
    """Tests for the handle_retry function."""

    def test_increments_retry_count(self, sample_agent_state):
        """Should increment retry count."""
        # Arrange
        sample_agent_state["retry_count"] = 0
        milestone = sample_agent_state["milestones"][0]
        validation = ValidationResult(
            success=False,
            reasoning="Failed",
            retry_strategy="Retry",
        )

        # Act
        result = auditor.handle_retry(sample_agent_state, milestone, validation)

        # Assert
        assert result["retry_count"] == 1

    def test_max_retries_resets_and_fails(self, sample_agent_state):
        """Should reset and mark failed after max retries."""
        # Arrange
        sample_agent_state["retry_count"] = 3  # Already at max
        milestone = sample_agent_state["milestones"][0]
        validation = ValidationResult(
            success=False,
            reasoning="Failed",
            retry_strategy=None,
        )

        # Act
        result = auditor.handle_retry(sample_agent_state, milestone, validation)

        # Assert
        assert result["retry_count"] == 0  # Reset
        assert result["last_verification_status"] == "failed"
        assert milestone.status == "failed"


class TestUpdateGlobalContext:
    """Tests for the update_global_context function."""

    def test_adds_milestone_summary(self):
        """Should add milestone summary to context."""
        # Arrange
        current_context = ""
        milestone = SubGoal(
            id="test",
            description="Test milestone",
            assigned_skill="test",
            status="completed",
        )
        history = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="tool", parameters={}, expected_outcome=""),
                result={},
                success=True,
            )
        ]

        # Act
        result = auditor.update_global_context(current_context, milestone, history)

        # Assert
        assert "Test milestone" in result
        assert "COMPLETED" in result
        assert "1 successful" in result

    def test_appends_to_existing_context(self):
        """Should append to existing context."""
        # Arrange
        current_context = "Previous milestone completed."
        milestone = SubGoal(
            id="test",
            description="New milestone",
            assigned_skill="test",
            status="completed",
        )
        history = []

        # Act
        result = auditor.update_global_context(current_context, milestone, history)

        # Assert
        assert "Previous milestone" in result
        assert "New milestone" in result


class TestCreateValidationPrompt:
    """Tests for the create_validation_prompt function."""

    def test_includes_milestone_details(self, sample_agent_state):
        """Prompt should include milestone description and criteria."""
        # Act
        prompt = auditor.create_validation_prompt(sample_agent_state)

        # Assert
        assert "List all Python files" in prompt
        assert "Success Criteria:" in prompt

    def test_includes_execution_history(self, sample_agent_state, sample_execution_record):
        """Prompt should include tool execution history."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]

        # Act
        prompt = auditor.create_validation_prompt(sample_agent_state)

        # Assert
        assert "TOOLS EXECUTED:" in prompt
