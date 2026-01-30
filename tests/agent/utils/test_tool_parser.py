"""Unit tests for the tool_parser utility module."""

import pytest

from agent.state import ExecutionRecord, ToolCall
from agent.utils.tool_parser import (
    format_errors,
    format_execution_context,
    format_execution_results,
    format_milestone_history,
    get_milestone_history,
    has_execution_failures,
    parse_tool_id,
)


class TestParseToolId:
    """Tests for parse_tool_id function."""

    def test_parses_valid_tool_id(self):
        """Should parse valid server:tool format."""
        server, tool = parse_tool_id("filesystem:list_files")
        assert server == "filesystem"
        assert tool == "list_files"

    def test_parses_tool_with_colon_in_name(self):
        """Should handle tool names with colons."""
        server, tool = parse_tool_id("server:tool:with:colons")
        assert server == "server"
        assert tool == "tool:with:colons"

    def test_raises_on_invalid_format(self):
        """Should raise ValueError for invalid format."""
        with pytest.raises(ValueError):
            parse_tool_id("invalid_tool_id")


class TestFormatExecutionContext:
    """Tests for format_execution_context function."""

    def test_formats_list_values(self):
        """Should format list values with item count."""
        context = {"files": ["a.py", "b.py", "c.py"]}
        result = format_execution_context(context)
        assert "files: 3 items" in result
        assert "a.py" in result

    def test_limits_list_display(self):
        """Should show only first 5 items for long lists."""
        context = {"files": [f"file{i}.py" for i in range(10)]}
        result = format_execution_context(context)
        assert "and 5 more" in result

    def test_formats_dict_values(self):
        """Should format dict values with indentation."""
        context = {"config": {"key1": "value1", "key2": "value2"}}
        result = format_execution_context(context)
        assert "config:" in result
        assert "key1: value1" in result

    def test_handles_empty_context(self):
        """Should handle empty context."""
        result = format_execution_context({})
        assert "No context accumulated" in result


class TestFormatMilestoneHistory:
    """Tests for format_milestone_history function."""

    def test_formats_successful_executions(self):
        """Should format successful tool executions."""
        records = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="tool1", parameters={}, expected_outcome=""),
                result={},
                success=True,
            ),
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="tool2", parameters={}, expected_outcome=""),
                result={},
                success=False,
            ),
        ]
        result = format_milestone_history(records)
        assert "SUCCESS" in result
        assert "FAILED" in result
        assert "tool1" in result
        assert "tool2" in result

    def test_handles_empty_history(self):
        """Should handle empty history."""
        result = format_milestone_history([])
        assert "No tools executed" in result


class TestFormatExecutionResults:
    """Tests for format_execution_results function."""

    def test_formats_successful_results(self):
        """Should format successful results with truncation."""
        records = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="tool", parameters={}, expected_outcome=""),
                result={"result": "a" * 300},  # Long result
                success=True,
            )
        ]
        result = format_execution_results(records)
        assert "tool:" in result
        assert len(result) < 400  # Should be truncated

    def test_formats_error_results(self):
        """Should format error results."""
        records = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="tool", parameters={}, expected_outcome=""),
                result={"error": "Connection failed"},
                success=False,
            )
        ]
        result = format_execution_results(records)
        assert "ERROR" in result
        assert "Connection failed" in result


class TestFormatErrors:
    """Tests for format_errors function."""

    def test_formats_error_list(self):
        """Should format list of errors."""
        errors = [
            {"error_message": "Error 1", "suggested_fix": "Fix 1"},
            {"error_message": "Error 2", "suggested_fix": "Fix 2"},
        ]
        result = format_errors(errors)
        assert "Previous attempts failed" in result
        assert "Error 1" in result
        assert "Fix 1" in result

    def test_limits_to_last_three(self):
        """Should only show last 3 errors."""
        errors = [{"error_message": f"Error {i}"} for i in range(5)]
        result = format_errors(errors)
        assert "Error 4" in result
        assert "Error 0" not in result

    def test_handles_empty_errors(self):
        """Should handle empty error list."""
        result = format_errors([])
        assert "No previous errors" in result


class TestGetMilestoneHistory:
    """Tests for get_milestone_history function."""

    def test_filters_by_milestone_id(self):
        """Should filter records by milestone ID."""
        records = [
            ExecutionRecord(
                milestone_id="m1",
                tool_call=ToolCall(tool_id="t1", parameters={}, expected_outcome=""),
                result={},
                success=True,
            ),
            ExecutionRecord(
                milestone_id="m2",
                tool_call=ToolCall(tool_id="t2", parameters={}, expected_outcome=""),
                result={},
                success=True,
            ),
        ]
        result = get_milestone_history(records, "m1")
        assert len(result) == 1
        assert result[0].milestone_id == "m1"


class TestHasExecutionFailures:
    """Tests for has_execution_failures function."""

    def test_detects_failures(self):
        """Should return True if any record failed."""
        records = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="t1", parameters={}, expected_outcome=""),
                result={},
                success=True,
            ),
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="t2", parameters={}, expected_outcome=""),
                result={},
                success=False,
            ),
        ]
        assert has_execution_failures(records) is True

    def test_returns_false_all_success(self):
        """Should return False if all succeeded."""
        records = [
            ExecutionRecord(
                milestone_id="test",
                tool_call=ToolCall(tool_id="t1", parameters={}, expected_outcome=""),
                result={},
                success=True,
            )
        ]
        assert has_execution_failures(records) is False
