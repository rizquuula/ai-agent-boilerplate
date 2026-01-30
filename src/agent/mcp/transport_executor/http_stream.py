from typing import Any

import requests

from .base import BaseTransport


class HTTPStreamTransport(BaseTransport):
    """Transport for MCP servers using HTTP streaming."""

    def __init__(self):
        self._session: requests.Session | None = None
        self._base_url: str | None = None
        self._timeout: int = 30

    def start(self, command: str, args: list[str]) -> None:
        """Initialize HTTP session with base URL."""
        if not args:
            raise ValueError("HTTP transport requires server URL in args")

        self._base_url = args[0]
        self._session = requests.Session()

        # Verify connection
        try:
            response = self._session.get(f"{self._base_url}/health", timeout=self._timeout)
            if not response.ok:
                raise RuntimeError(f"Server health check failed: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"HTTP connection failed: {str(e)}") from e

    def stop(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
        self._base_url = None

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via HTTP streaming."""
        if not self.is_alive():
            raise RuntimeError("HTTP transport is not connected")

        url = f"{self._base_url}/tools/{tool_name}"

        try:
            with self._session.post(url, json=kwargs, stream=True, timeout=self._timeout) as response:
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"HTTP error {response.status_code}: {response.text}",
                    }

                import json

                result = {}
                for chunk in response.iter_lines():
                    if chunk:
                        result.update(json.loads(chunk))

                return {"success": True, "result": result}

        except requests.RequestException as e:
            return {"success": False, "error": f"HTTP request failed: {str(e)}"}

    def list_tools(self) -> list[str]:
        """List available tools via HTTP endpoint."""
        try:
            response = self.execute_tool("_list_tools")
            if response["success"]:
                return response["result"].get("tools", [])
            return []
        except Exception:
            return []

    def is_alive(self) -> bool:
        """Check if HTTP session is active."""
        return self._session is not None and self._base_url is not None
