"""Test MCP executor system."""

from unittest.mock import MagicMock, patch

import pytest

from agent.mcp.executor import MCPExecutor, execute_mcp_tool, get_mcp_executor


@pytest.fixture
def mock_config():
    """Create a mock MCPConfig."""
    config = MagicMock()
    config.get_server_metadata.return_value = {
        "command": "test_command",
        "args": ["arg1", "arg2"],
        "transport": "stdio",
    }
    config.is_server_enabled.return_value = True
    config.get_enabled_servers.return_value = ["filesystem", "code_parser"]
    return config


@pytest.fixture
def mock_transport():
    """Create a mock transport."""
    transport = MagicMock()
    transport.list_tools.return_value = ["list_files", "read_file", "write_file", "get_file_info"]
    transport.execute_tool.return_value = {"success": True, "data": "test_result"}
    transport.is_alive.return_value = True
    return transport


class TestMCPExecutor:
    """Test cases for MCPExecutor class."""

    def test_mcp_executor_initialization_with_default_config(self):
        """Test MCP executor initialization with default config."""
        with patch("agent.mcp.executor.get_mcp_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            executor = MCPExecutor()

            assert executor.config is mock_config
            assert executor.transports == {}
            assert executor.tool_cache == {}

    def test_mcp_executor_initialization_with_custom_config(self):
        """Test MCP executor initialization with custom config path."""
        with patch("agent.mcp.executor.MCPConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            executor = MCPExecutor("/custom/path/config.json")

            mock_config_class.assert_called_once_with("/custom/path/config.json")
            mock_config.load_config.assert_called_once()
            assert executor.config is mock_config

    def test_execute_tool_valid_server(self, mock_config, mock_transport):
        """Test executing tools on valid servers."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                result = executor.execute_tool("filesystem", "list_files")

                assert result["success"] is True
                assert result["tool"] == "filesystem:list_files"
                assert result["tool_call"] == "filesystem:list_files"
                assert result["error"] is None
                mock_transport.start.assert_called_once_with("test_command", ["arg1", "arg2"])
                mock_transport.list_tools.assert_called_once()
                mock_transport.execute_tool.assert_called_once_with("list_files")

    def test_execute_tool_invalid_server(self, mock_config):
        """Test executing tools on invalid servers."""
        mock_config.get_server_metadata.return_value = None

        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            executor = MCPExecutor()

            result = executor.execute_tool("invalid_server", "list_files")

            assert result["success"] is False
            assert "error" in result
            assert "invalid_server" in result["error"]

    def test_execute_tool_disabled_server(self, mock_config):
        """Test executing tools on disabled servers."""
        mock_config.is_server_enabled.return_value = False

        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            executor = MCPExecutor()

            result = executor.execute_tool("filesystem", "list_files")

            assert result["success"] is False
            assert "error" in result
            assert "not enabled" in result["error"]

    def test_execute_tool_invalid_tool(self, mock_config, mock_transport):
        """Test executing invalid tools."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                # Cache the tools first
                executor._get_transport("filesystem")

                result = executor.execute_tool("filesystem", "invalid_tool")

                assert result["success"] is False
                assert "error" in result
                assert "invalid_tool" in result["error"]

    def test_get_available_tools(self, mock_config, mock_transport):
        """Test getting available tools."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()
                available_tools = executor.get_available_tools()

                assert "filesystem" in available_tools
                assert "code_parser" in available_tools
                assert available_tools["filesystem"] == ["list_files", "read_file", "write_file", "get_file_info"]

    def test_get_available_tools_with_exception(self, mock_config):
        """Test getting available tools when transport fails."""
        mock_config.get_server_metadata.side_effect = [
            {"command": "test", "args": [], "transport": "stdio"},
            None,  # Second server fails
        ]

        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", side_effect=Exception("Connection failed")):
                executor = MCPExecutor()
                available_tools = executor.get_available_tools()

                assert available_tools == {"filesystem": [], "code_parser": []}

    def test_validate_tool_call(self, mock_config, mock_transport):
        """Test tool call validation."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                # Cache the tools first
                executor._get_transport("filesystem")

                # Valid tool calls
                assert executor.validate_tool_call("filesystem", "list_files") is True
                assert executor.validate_tool_call("filesystem", "read_file") is True

                # Invalid tool calls
                assert executor.validate_tool_call("filesystem", "invalid_tool") is False
                assert executor.validate_tool_call("invalid_server", "list_files") is False

    def test_shutdown(self, mock_config, mock_transport):
        """Test shutdown cleans up transports."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                # Initialize a transport
                executor._get_transport("filesystem")

                executor.shutdown()

                mock_transport.stop.assert_called_once()
                assert executor.transports == {}
                assert executor.tool_cache == {}

    def test_execute_tool_with_parameters(self, mock_config, mock_transport):
        """Test tool execution with parameters."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                result = executor.execute_tool("filesystem", "list_files", pattern="*.py", directory="/test")

                assert result["success"] is True
                mock_transport.execute_tool.assert_called_once_with("list_files", pattern="*.py", directory="/test")

    def test_execute_tool_transport_error(self, mock_config, mock_transport):
        """Test error handling when transport fails."""
        mock_transport.execute_tool.side_effect = Exception("Connection lost")

        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                executor = MCPExecutor()

                result = executor.execute_tool("filesystem", "list_files")

                assert result["success"] is False
                assert "error" in result
                assert "Connection lost" in result["error"]


class TestGlobalFunctions:
    """Test cases for global executor functions."""

    def test_get_mcp_executor_singleton(self):
        """Test that get_mcp_executor returns a singleton instance."""
        with patch("agent.mcp.executor.MCPExecutor") as mock_executor_class:
            mock_instance = MagicMock()
            mock_executor_class.return_value = mock_instance

            # Reset the global instance
            import agent.mcp.executor as executor_module

            executor_module._mcp_executor = None

            result1 = get_mcp_executor()
            result2 = get_mcp_executor()

            assert result1 is result2
            mock_executor_class.assert_called_once()

    def test_execute_mcp_tool_function(self, mock_config, mock_transport):
        """Test the standalone execute_mcp_tool function."""
        with patch("agent.mcp.executor.get_mcp_config", return_value=mock_config):
            with patch("agent.mcp.executor.create_transport", return_value=mock_transport):
                # Reset the global instance to ensure fresh executor
                import agent.mcp.executor as executor_module

                executor_module._mcp_executor = None

                # Test valid tool
                result = execute_mcp_tool("filesystem", "list_files")

                assert result["success"] is True
                assert result["tool"] == "filesystem:list_files"


if __name__ == "__main__":
    pytest.main([__file__])
