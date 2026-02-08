"""Task input resolution based on execution history."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import LLMUsage, Task, TaskInputResolverResult
from asterism.agent.nodes.shared import LLMCaller, get_user_request, has_execution_history
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

RESOLVER_SYSTEM_PROMPT = """You are a task input resolver.
Extract information from previous results to update task inputs.
Return only valid JSON in the specified format."""


def resolve_next_task_inputs(
    llm: BaseLLMProvider,
    next_task: Task,
    state: AgentState,
) -> tuple[dict[str, Any] | None, LLMUsage | None]:
    """Resolve task inputs by analyzing execution history using LLM.

    Args:
        llm: The LLM provider for resolution.
        next_task: The task whose inputs need resolution.
        state: Current agent state with execution history.

    Returns:
        Tuple of (updated_tool_input or None, LLMUsage or None).
    """
    if not has_execution_history(state):
        return None, None

    caller = LLMCaller(llm, "task_resolver")
    messages = _build_resolver_messages(next_task, state)

    try:
        result = caller.call_structured(
            messages,
            TaskInputResolverResult,
            f"resolving inputs for task {next_task.id}",
        )

        return result.parsed.updated_tool_input, result.usage

    except Exception:
        # If resolution fails, return None to use original inputs
        return None, None


def _build_resolver_messages(task: Task, state: AgentState) -> list:
    """Build messages for the resolver LLM.

    Args:
        task: The task to resolve inputs for.
        state: Current agent state.

    Returns:
        List of LangChain messages.
    """
    user_request = get_user_request(state)
    execution_results = state.get("execution_results", [])
    current_input = task.tool_input or {}

    # Format execution history
    history_lines = []
    for i, result in enumerate(execution_results):
        status = "✓" if result.success else "✗"
        result_str = str(result.result) if result.result else "None"
        history_lines.append(f"\nTask {i + 1} ({result.task_id}): {status}\nResult: {result_str}")

    history = "".join(history_lines)

    user_prompt = f"""=== USER REQUEST ===
{user_request}

=== EXECUTION HISTORY ===
{history}

=== NEXT TASK TO RESOLVE ===
Task ID: {task.id}
Description: {task.description}
Tool: {task.tool_call}
Current tool_input: {current_input}

=== TASK ===
Analyze the execution history and update the tool_input for the next task.

The previous task results may contain information needed for this task
(e.g., file paths, IDs, search results).

Instructions:
1. Review the execution history to find relevant information
2. Update the tool_input with actual values from previous results
3. Return ONLY a JSON object with the updated_tool_input field

Example:
If a search returned "/path/to/file.txt", and the next task needs to read it:
Current: {{"path": "file.txt"}}
Updated: {{"path": "/path/to/file.txt"}}

Return format:
{{"updated_tool_input": {{...}}}}

If no updates needed, return: {{"updated_tool_input": null}}"""

    return [
        SystemMessage(content=RESOLVER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
