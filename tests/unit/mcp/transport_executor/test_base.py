"""Test MCP base transport abstract class."""

from typing import Any
from unittest.mock import patch

import pytest

from asterism.mcp.transport_executor.base import BaseTransport


def test_base_transport_cannot_be_instantiated():
    """Test that BaseTransport cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTransport()


def test_base_transport_abstract_methods():
    """Test that all abstract methods must be implemented."""

    class IncompleteTransport(BaseTransport):
        pass

    with pytest.raises(TypeError):
        IncompleteTransport()


def test_base_transport_partial_implementation():
    """Test that partial implementation still raises TypeError."""

    class PartialTransport(BaseTransport):
        def start(self, command: str, args: list[str]) -> None:
            pass

        def stop(self) -> None:
            pass

    with pytest.raises(TypeError):
        PartialTransport()


@patch("asterism.mcp.transport_executor.base.BaseTransport.__abstractmethods__", set())
def test_base_transport_method_signatures():
    """Test that BaseTransport defines correct method signatures."""
    # Create a concrete implementation for testing signatures
    transport = BaseTransport()

    # Test method existence
    assert hasattr(transport, "start")
    assert hasattr(transport, "stop")
    assert hasattr(transport, "execute_tool")
    assert hasattr(transport, "list_tools")
    assert hasattr(transport, "is_alive")


class ConcreteTransport(BaseTransport):
    """A concrete implementation of BaseTransport for testing."""

    def __init__(self):
        self._running = False

    def start(self, command: str, args: list[str]) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def execute_tool(self, tool_name: str, **kwargs):
        return {"success": True, "result": f"Executed {tool_name}"}

    def list_tools(self) -> list[str]:
        return ["tool1", "tool2"]
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [{}]

    def is_alive(self) -> bool:
        return self._running


def test_concrete_transport_implementation():
    """Test that a fully implemented transport works correctly."""
    transport = ConcreteTransport()

    # Test initial state
    assert not transport.is_alive()

    # Test start
    transport.start("test", ["arg1"])
    assert transport.is_alive()

    # Test execute_tool
    result = transport.execute_tool("test_tool")
    assert result["success"] is True
    assert "test_tool" in result["result"]

    # Test list_tools
    tools = transport.list_tools()
    assert "tool1" in tools
    assert "tool2" in tools

    # Test stop
    transport.stop()
    assert not transport.is_alive()


if __name__ == "__main__":
    pytest.main([__file__])
