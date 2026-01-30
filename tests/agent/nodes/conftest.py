"""Test fixtures for agent node tests."""

from unittest.mock import Mock

import pytest

from agent.state import (
    ErrorRecord,
    ExecutionRecord,
    SubGoal,
    ToolCall,
)


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock()
    provider.name = "mock"
    provider.model = "mock-model"
    return provider


@pytest.fixture
def mock_mcp_executor():
    """Mock MCP executor for testing."""
    return Mock()


@pytest.fixture
def sample_milestone():
    """Sample milestone for testing."""
    return SubGoal(
        id="test-milestone-123",
        description="List all Python files in the project",
        assigned_skill="filesystem",
        status="pending",
        success_criteria="At least one .py file is found or directory is confirmed empty",
    )


@pytest.fixture
def sample_agent_state(sample_milestone):
    """Sample agent state for testing."""
    return {
        "objective": "Analyze Python project structure",
        "milestones": [sample_milestone],
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


@pytest.fixture
def sample_tool_call():
    """Sample tool call for testing."""
    return ToolCall(
        tool_id="filesystem:list_files",
        parameters={"path": ".", "pattern": "*.py"},
        expected_outcome="List of Python files in root directory",
    )


@pytest.fixture
def sample_execution_record(sample_milestone, sample_tool_call):
    """Sample execution record for testing."""
    return ExecutionRecord(
        milestone_id=sample_milestone.id,
        tool_call=sample_tool_call,
        result={"success": True, "result": ["main.py", "utils.py"]},
        success=True,
    )


@pytest.fixture
def sample_error_record(sample_milestone):
    """Sample error record for testing."""
    return ErrorRecord(
        milestone_id=sample_milestone.id,
        error_message="Tool execution failed",
        suggested_fix="Check file permissions and retry",
    )


@pytest.fixture
def completed_milestone():
    """Completed milestone for testing."""
    return SubGoal(
        id="completed-milestone-456",
        description="Find configuration files",
        assigned_skill="filesystem",
        status="completed",
        success_criteria="Config files are located",
    )


@pytest.fixture
def state_with_history(sample_agent_state, sample_execution_record):
    """State with execution history for testing."""
    state = sample_agent_state.copy()
    state["history"] = [sample_execution_record]
    return state


@pytest.fixture
def state_with_errors(sample_agent_state, sample_error_record):
    """State with errors for testing."""
    state = sample_agent_state.copy()
    state["errors"] = [sample_error_record]
    state["retry_count"] = 1
    return state
