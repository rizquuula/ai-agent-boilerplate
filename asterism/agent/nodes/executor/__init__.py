"""Executor node package.

Executes tasks from the plan using MCP tools or LLM processing.
"""

from .node import executor_node

__all__ = ["executor_node"]
