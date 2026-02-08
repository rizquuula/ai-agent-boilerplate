"""Unit tests for StdioTransport."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from asterism.mcp.transport_executor.stdio import StdioTransport


def test_stdio_transport_init():
    """Test StdioTransport initialization."""
    transport = StdioTransport()
    assert transport._process is None
    assert transport._request_id == 0
    assert transport._initialized is False


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_start_success(mock_popen_class):
    """Test successful start and initialization."""
    # Setup mock process
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    )
    mock_process.poll.return_value = None
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    mock_popen_class.assert_called_once_with(
        ["python", "-m", "test_server"],
        stdin=-1,
        stdout=-1,
        stderr=-1,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=None,
    )
    assert transport._initialized is True
    assert transport._process is not None


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_execute_tool_success(mock_popen_class):
    """Test successful tool execution."""
    # Setup mock process
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    # First call for initialization, second for tool execution
    mock_process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": [{"type": "text", "text": json.dumps({"result": "success"})}]},
            }
        ),
    ]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    result = transport.execute_tool("test_tool", param1="value1")

    assert result == {"result": "success"}


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_list_tools_success(mock_popen_class):
    """Test successful tools listing."""
    # Setup mock process
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    # First call for initialization, second for list_tools
    mock_process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": [{"name": "tool1"}, {"name": "tool2"}]},
            }
        ),
    ]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    tools = transport.list_tools()

    assert "tool1" in tools
    assert "tool2" in tools


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_is_alive(mock_popen_class):
    """Test is_alive returns correct state."""
    transport = StdioTransport()
    assert not transport.is_alive()

    # Setup mock process
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    )
    mock_process.poll.return_value = None
    mock_popen_class.return_value = mock_process

    transport.start("python", ["-m", "test_server"])
    assert transport.is_alive()

    mock_process.poll.return_value = 0  # Process exited
    assert not transport.is_alive()


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_stop_terminates_process(mock_popen_class):
    """Test stop terminates the process."""
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    )
    mock_process.poll.return_value = None
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])
    transport.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)
    assert transport._initialized is False


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_stop_kills_process_if_needed(mock_popen_class):
    """Test stop kills process if terminate times out."""
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    )
    mock_process.poll.return_value = None
    mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=5), None]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])
    transport.stop()

    mock_process.terminate.assert_called_once()
    mock_process.kill.assert_called_once()


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_execute_tool_python_literal_parsing(mock_popen_class):
    """Test tool execution with Python literal fallback parsing."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    # First call for initialization, second for tool execution with Python literal
    mock_process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": [{"type": "text", "text": "{'key': 'value'}"}]},
            }
        ),
    ]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    result = transport.execute_tool("test_tool")

    assert result == {"key": "value"}


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_execute_tool_empty_content(mock_popen_class):
    """Test tool execution with empty content returns empty dict."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    mock_process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": []},
            }
        ),
    ]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    result = transport.execute_tool("test_tool")

    assert result == {}


@patch("asterism.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_list_tools_returns_tool_names(mock_popen_class):
    """Test list_tools returns list of tool names."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    mock_process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "get_time", "description": "Get current time"},
                        {"name": "get_date", "description": "Get current date"},
                    ]
                },
            }
        ),
    ]
    mock_popen_class.return_value = mock_process

    transport = StdioTransport()
    transport.start("python", ["-m", "test_server"])

    tools = transport.list_tools()

    assert len(tools) == 2
    assert "get_time" in tools
    assert "get_date" in tools


if __name__ == "__main__":
    pytest.main([__file__])
