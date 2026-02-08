"""Context building for the planner node."""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.state import AgentState
from asterism.agent.utils import get_workspace_tree_context
from asterism.mcp.executor import MCPExecutor

from .prompts import PLANNER_SYSTEM_PROMPT
from .utils import format_tools_context


@dataclass
class PlannerContext:
    """Context container for planner node.

    Encapsulates all data needed for planning:
    - messages: LLM messages to send
    - user_message: Original user request
    - tools_context: Formatted tool descriptions
    - workspace_context: Workspace tree information
    """

    messages: list
    user_message: str
    tools_context: str
    workspace_context: str


def build_planner_context(
    state: AgentState,
    mcp_executor: MCPExecutor,
    workspace_root: str,
) -> PlannerContext:
    """Build complete planning context from state.

    Args:
        state: Current agent state.
        mcp_executor: MCP executor for tool discovery.
        workspace_root: Path to workspace for context.

    Returns:
        PlannerContext with all necessary information.
    """
    user_message = _extract_user_message(state)
    execution_context = _build_execution_context(state)
    tools_context = _fetch_tools_context(mcp_executor)
    workspace_context = get_workspace_tree_context(workspace_root)

    messages = _build_messages(
        user_message=user_message,
        execution_context=execution_context,
        tools_context=tools_context,
        workspace_context=workspace_context,
    )

    return PlannerContext(
        messages=messages,
        user_message=user_message,
        tools_context=tools_context,
        workspace_context=workspace_context,
    )


def _extract_user_message(state: AgentState) -> str:
    """Extract the most recent user message from state."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_execution_context(state: AgentState) -> str:
    """Build context string from execution history."""
    results = state.get("execution_results", [])
    if not results:
        return ""

    lines = ["\n\nExecution History:"]
    for result in results:
        status = "✓" if result.success else "✗"
        content = result.result if result.success else result.error
        lines.append(f"- {status} {result.task_id}: {content}")

    return "\n".join(lines)


def _fetch_tools_context(mcp_executor: MCPExecutor) -> str:
    """Fetch and format available tools from MCP."""
    try:
        tool_schemas = mcp_executor.get_tool_schemas()
        return format_tools_context(tool_schemas)
    except Exception as e:
        return f"Error getting tool information: {str(e)}"


def _build_messages(
    user_message: str,
    execution_context: str,
    tools_context: str,
    workspace_context: str,
) -> list:
    """Build LLM messages for planning.

    Args:
        user_message: The user's request.
        execution_context: Previous execution results (if any).
        tools_context: Formatted tool descriptions.
        workspace_context: Workspace tree info.

    Returns:
        List of LangChain messages.
    """
    system_prompt = _build_system_prompt(tools_context, workspace_context)

    user_prompt = f"""User Request: {user_message}

{execution_context}

Create a plan to accomplish this request using the available tools.

JSON OUTPUT:"""

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def _build_system_prompt(tools_context: str, workspace_context: str) -> str:
    """Build the enhanced system prompt with context."""
    return f"""{PLANNER_SYSTEM_PROMPT}

{workspace_context}

Available MCP Tools:
{tools_context}

When creating tasks with tool calls:
- Use ONLY tools listed above
- Use exact format: "server_name:tool_name"
- Provide all required parameters in tool_input
- If a tool is not available, use LLM reasoning instead
"""
