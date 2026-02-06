"""Fixtures for agent unit tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.name = "MockLLM"
    llm.model = "mock-model"

    # Default invoke returns simple string
    llm.invoke.return_value = "Mock LLM response"

    # Default invoke_structured returns appropriate model
    def mock_invoke_structured(prompt, schema, **kwargs):
        from asterism.agent.models import EvaluationDecision, EvaluationResult, Plan, Task

        if schema.__name__ == "Plan":
            return Plan(
                tasks=[
                    Task(
                        id="task_1",
                        description="Mock task",
                        tool_call=None,
                        tool_input=None,
                        depends_on=[],
                    )
                ],
                reasoning="Mock reasoning",
            )
        elif schema.__name__ == "EvaluationResult":
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                reasoning="Mock evaluation - execution on track",
            )
        return MagicMock()

    llm.invoke_structured.side_effect = mock_invoke_structured
    return llm


@pytest.fixture
def mock_mcp_executor():
    """Create a mock MCP executor."""
    from unittest.mock import MagicMock

    executor = MagicMock()
    executor.execute_tool.return_value = {
        "success": True,
        "result": "Mock tool result",
        "error": None,
        "tool": "test:tool",
        "tool_call": "test:tool",
    }
    executor.get_available_tools.return_value = {"test": ["tool"]}
    executor.validate_tool_call.return_value = True
    executor.get_tool_schemas.return_value = {
        "test": [
            {
                "name": "tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string", "description": "A parameter"}
                    },
                    "required": ["param"],
                },
            }
        ]
    }
    return executor
