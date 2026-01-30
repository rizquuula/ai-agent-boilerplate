"""Unit tests for the executor node."""

from agent.nodes import executor
from agent.state import ExecutionRecord, ToolCall


class TestExecutorNode:
    """Tests for the executor node."""

    def test_executor_runs_tactical_plan(self, mock_mcp_executor, sample_agent_state, sample_tool_call):
        """Should execute each ToolCall in plan."""
        # Arrange
        sample_agent_state["tactical_plan"] = [
            sample_tool_call,
            ToolCall(
                tool_id="filesystem:read_file",
                parameters={"path": "test.py"},
                expected_outcome="Read file",
            ),
        ]
        mock_mcp_executor.execute_tool.return_value = {
            "success": True,
            "result": ["file1.py", "file2.py"],
        }

        # Act
        result = executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        assert mock_mcp_executor.execute_tool.call_count == 2
        assert len(result["history"]) == 2

    def test_executor_records_results(self, mock_mcp_executor, sample_agent_state, sample_tool_call):
        """Should add ExecutionRecord to history."""
        # Arrange
        sample_agent_state["tactical_plan"] = [sample_tool_call]
        mock_result = {
            "success": True,
            "result": ["main.py"],
        }
        mock_mcp_executor.execute_tool.return_value = mock_result

        # Act
        result = executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        assert len(result["history"]) == 1
        assert isinstance(result["history"][0], ExecutionRecord)
        assert result["history"][0].success is True
        assert result["history"][0].result == mock_result
        assert result["history"][0].tool_call == sample_tool_call

    def test_executor_handles_tool_failure(self, mock_mcp_executor, sample_agent_state, sample_tool_call):
        """Should record failure without crashing."""
        # Arrange
        sample_agent_state["tactical_plan"] = [sample_tool_call]
        mock_mcp_executor.execute_tool.side_effect = Exception("Connection failed")

        # Act
        result = executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        assert len(result["history"]) == 1
        assert result["history"][0].success is False
        assert "Connection failed" in str(result["history"][0].result.get("error", ""))

    def test_executor_parses_tool_id_correctly(self, mock_mcp_executor, sample_agent_state):
        """Should correctly parse server:tool format."""
        # Arrange
        sample_agent_state["tactical_plan"] = [
            ToolCall(
                tool_id="code_parser:analyze_complexity",
                parameters={"file": "test.py"},
                expected_outcome="Analyze",
            )
        ]
        mock_mcp_executor.execute_tool.return_value = {"success": True, "result": {}}

        # Act
        executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        call_kwargs = mock_mcp_executor.execute_tool.call_args.kwargs
        assert call_kwargs.get("server_name") == "code_parser"
        assert call_kwargs.get("tool_name") == "analyze_complexity"

    def test_executor_handles_no_active_milestone(self, mock_mcp_executor):
        """Should handle state with no milestones."""
        # Arrange
        state = {
            "milestones": [],
            "current_idx": 0,
            "tactical_plan": [ToolCall(tool_id="tool", parameters={}, expected_outcome="")],
            "history": [],
        }

        # Act
        result = executor.node(state, mock_mcp_executor)

        # Assert
        mock_mcp_executor.execute_tool.assert_not_called()
        assert result["history"] == []

    def test_executor_preserves_existing_history(
        self, mock_mcp_executor, sample_agent_state, sample_execution_record, sample_tool_call
    ):
        """Should append to existing history."""
        # Arrange
        sample_agent_state["history"] = [sample_execution_record]
        sample_agent_state["tactical_plan"] = [sample_tool_call]
        mock_mcp_executor.execute_tool.return_value = {"success": True, "result": {}}

        # Act
        result = executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        assert len(result["history"]) == 2
        assert result["history"][0] == sample_execution_record

    def test_executor_passes_parameters_correctly(self, mock_mcp_executor, sample_agent_state):
        """Should pass parameters to MCP executor correctly."""
        # Arrange
        sample_agent_state["tactical_plan"] = [
            ToolCall(
                tool_id="filesystem:list_files",
                parameters={
                    "path": "/tmp",
                    "pattern": "*.py",
                    "recursive": True,
                },
                expected_outcome="List files",
            )
        ]
        mock_mcp_executor.execute_tool.return_value = {"success": True, "result": []}

        # Act
        executor.node(sample_agent_state, mock_mcp_executor)

        # Assert
        call_kwargs = mock_mcp_executor.execute_tool.call_args.kwargs
        assert call_kwargs.get("path") == "/tmp"
        assert call_kwargs.get("pattern") == "*.py"
        assert call_kwargs.get("recursive") is True


class TestUpdateExecutionContext:
    """Tests for the update_execution_context function."""

    def test_updates_with_file_list_results(self, sample_agent_state, sample_tool_call):
        """Should update context with discovered files."""
        # Arrange
        result = {"success": True, "result": ["main.py", "utils.py"]}
        sample_tool_call.tool_id = "filesystem:list_files"

        # Act
        updated_context = executor.update_execution_context(sample_agent_state, sample_tool_call, result)

        # Assert
        assert "discovered_files" in updated_context
        assert "main.py" in updated_context["discovered_files"]

    def test_updates_with_file_contents(self, sample_agent_state, sample_tool_call):
        """Should update context with file contents."""
        # Arrange
        result = {"success": True, "result": "file content here"}
        sample_tool_call.tool_id = "filesystem:read_file"
        sample_tool_call.parameters = {"path": "test.py"}

        # Act
        updated_context = executor.update_execution_context(sample_agent_state, sample_tool_call, result)

        # Assert
        assert "file_contents" in updated_context
        assert "test.py" in updated_context["file_contents"]

    def test_skips_failed_results(self, sample_agent_state, sample_tool_call):
        """Should not update context for failed results."""
        # Arrange
        result = {"success": False, "error": "Failed"}
        sample_tool_call.tool_id = "filesystem:list_files"

        # Act
        updated_context = executor.update_execution_context(sample_agent_state, sample_tool_call, result)

        # Assert
        assert updated_context == {}

    def test_preserves_existing_context(self, sample_agent_state, sample_tool_call):
        """Should preserve existing context entries."""
        # Arrange
        sample_agent_state["execution_context"] = {"existing": "data"}
        result = {"success": True, "result": ["file.py"]}
        sample_tool_call.tool_id = "filesystem:list_files"

        # Act
        updated_context = executor.update_execution_context(sample_agent_state, sample_tool_call, result)

        # Assert
        assert updated_context["existing"] == "data"
        assert "discovered_files" in updated_context
