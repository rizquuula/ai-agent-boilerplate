from typing import Literal

from .base import BaseTransport
from .http_stream import HTTPStreamTransport
from .sse import SSETransport
from .stdio import StdioTransport

__all__ = ["BaseTransport", "StdioTransport", "SSETransport", "HTTPStreamTransport"]


def create_transport(transport_type: Literal["stdio", "sse", "http_stream"]) -> BaseTransport:
    """Factory function to create transport instances."""
    if transport_type == "stdio":
        return StdioTransport()
    elif transport_type == "sse":
        return SSETransport()
    elif transport_type == "http_stream":
        return HTTPStreamTransport()
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")
