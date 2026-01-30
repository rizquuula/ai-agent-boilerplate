"""Test MCP stdio transport."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent.mcp.transport_executor.stdio import StdioTransport


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_start_success(mock_popen):
    """Test successful start of stdio transport."""
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", ["arg1", "arg2"])

    mock_popen.assert_called_once_with(
        ["test_command", "arg1", "arg2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert transport._process == mock_process


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_start_failure(mock_popen):
    """Test start failure of stdio transport."""
    mock_popen.side_effect = Exception("Failed to start process")

    transport = StdioTransport()
    with pytest.raises(RuntimeError, match="Failed to start MCP server"):
        transport.start("test_command", ["arg1"])


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_stop_terminate(mock_popen):
    """Test stopping transport with terminate."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    transport.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_stop_kill(mock_popen):
    """Test stopping transport with kill after timeout."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    transport.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)
    mock_process.kill.assert_called_once()


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_stop_not_running(mock_popen):
    """Test stopping transport when not running."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Process already exited
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    transport.stop()

    mock_process.terminate.assert_not_called()


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_is_alive_true(mock_popen):
    """Test is_alive returns True when process is running."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])

    assert transport.is_alive() is True


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_is_alive_false(mock_popen):
    """Test is_alive returns False when process is not running."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])

    assert transport.is_alive() is False


def test_stdio_transport_is_alive_no_process():
    """Test is_alive returns False when no process exists."""
    transport = StdioTransport()
    assert transport.is_alive() is False


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_execute_tool_success(mock_popen):
    """Test successful tool execution."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout.readline.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "result": "test_result",
            "id": 1,
        }
    )
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    result = transport.execute_tool("test_tool", arg1="value1")

    assert result["result"] == "test_result"
    mock_process.stdin.write.assert_called_once()
    mock_process.stdin.flush.assert_called_once()


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_execute_tool_not_running(mock_popen):
    """Test tool execution when process is not running."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])

    with pytest.raises(RuntimeError, match="Server process is not running"):
        transport.execute_tool("test_tool")


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_execute_tool_failure(mock_popen):
    """Test tool execution failure."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdin.write.side_effect = Exception("Write failed")
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])

    with pytest.raises(RuntimeError, match="Tool execution failed"):
        transport.execute_tool("test_tool")


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_execute_tool_incrementing_id(mock_popen):
    """Test that request IDs are incrementing."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout.readline.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "result": "test",
            "id": 1,
        }
    )
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])

    # Execute tool multiple times
    transport.execute_tool("tool1")
    assert transport._request_id == 1

    transport.execute_tool("tool2")
    assert transport._request_id == 2

    transport.execute_tool("tool3")
    assert transport._request_id == 3


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_list_tools(mock_popen):
    """Test listing tools."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout.readline.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "result": ["tool1", "tool2", "tool3"],
            "id": 1,
        }
    )
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    tools = transport.list_tools()

    assert "tool1" in tools
    assert "tool2" in tools
    assert "tool3" in tools


@patch("agent.mcp.transport_executor.stdio.subprocess.Popen")
def test_stdio_transport_list_tools_empty(mock_popen):
    """Test listing tools returns empty list when no tools."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout.readline.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "result": [],
            "id": 1,
        }
    )
    mock_popen.return_value = mock_process

    transport = StdioTransport()
    transport.start("test_command", [])
    tools = transport.list_tools()

    assert tools == []


if __name__ == "__main__":
    pytest.main([__file__])
