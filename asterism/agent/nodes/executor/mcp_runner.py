"""MCP tool execution runner."""

import time
from dataclasses import dataclass
from typing import Any

from asterism.agent.models import TaskResult
from asterism.agent.nodes.executor.utils import parse_tool_call
from asterism.agent.state import AgentState
from asterism.agent.utils import log_mcp_tool_call
from asterism.mcp.executor import MCPExecutor


@dataclass
class MCPResult:
    """Result from MCP tool execution."""

    success: bool
    data: Any
    error: str | None
    duration_ms: float


class MCPRunner:
    """Runner for MCP tool execution tasks."""

    def __init__(self, executor: MCPExecutor):
        self.executor = executor
        self._logger = __import__("logging").getLogger(__name__)

    def execute(self, task, state: AgentState) -> TaskResult:
        """Execute an MCP tool call task.

        Args:
            task: Task with tool_call and tool_input.
            state: Current agent state (unused for MCP tasks).

        Returns:
            TaskResult with execution outcome.
        """
        server_name, tool_name = parse_tool_call(task.tool_call)
        tool_input = task.tool_input or {}

        self._logger.debug(f"MCP tool call: {server_name}:{tool_name}, input_keys: {list(tool_input.keys())}")

        mcp_result = self._execute_tool(server_name, tool_name, tool_input, task.id)

        return TaskResult(
            task_id=task.id,
            success=mcp_result.success,
            result=mcp_result.data if mcp_result.success else None,
            error=mcp_result.error if not mcp_result.success else None,
        )

    def _execute_tool(
        self,
        server_name: str,
        tool_name: str,
        tool_input: dict,
        task_id: str,
    ) -> MCPResult:
        """Execute the MCP tool with timing and logging.

        Args:
            server_name: MCP server name.
            tool_name: Tool name.
            tool_input: Tool parameters.
            task_id: Task ID for logging.

        Returns:
            MCPResult with execution outcome.
        """
        start_time = time.perf_counter()

        try:
            result = self.executor.execute_tool(server_name, tool_name, **tool_input)
            duration_ms = (time.perf_counter() - start_time) * 1000

            success = result.get("success", False)

            self._log_result(
                server_name=server_name,
                tool_name=tool_name,
                tool_input=tool_input,
                success=success,
                duration_ms=duration_ms,
                result=result,
            )

            return MCPResult(
                success=success,
                data=result.get("result") if success else None,
                error=result.get("error") if not success else None,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self._log_result(
                server_name=server_name,
                tool_name=tool_name,
                tool_input=tool_input,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
            )

            return MCPResult(
                success=False,
                data=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _log_result(
        self,
        server_name: str,
        tool_name: str,
        tool_input: dict,
        success: bool,
        duration_ms: float,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Log MCP tool execution result."""
        log_mcp_tool_call(
            logger=self._logger,
            server_name=server_name,
            tool_name=tool_name,
            input_keys=list(tool_input.keys()),
            success=success,
            duration_ms=duration_ms,
            result_preview=str(result.get("result", ""))[:500] if result else None,
            error=error,
        )
