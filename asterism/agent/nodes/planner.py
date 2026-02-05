"""Planner node for creating and updating task plans."""

from langchain_core.messages import HumanMessage

from asterism.agent.models import Plan
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider


def _generate_task_id(index: int, description: str) -> str:
    """Generate a unique task ID."""
    return f"task_{index}_{description.lower().replace(' ', '_')[:30]}"


def planner_node(llm: BaseLLMProvider, state: AgentState) -> AgentState:
    """
    Create or update a plan based on the user request and execution history.

    Args:
        llm: The LLM provider to use for planning.
        state: Current agent state.

    Returns:
        Updated state with a new or updated plan.
    """
    # Build context from messages and execution results
    user_message = ""
    if state["messages"]:
        # Get the latest user message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

    execution_context = ""
    if state["execution_results"]:
        execution_context = "\n\nExecution History:\n"
        for result in state["execution_results"]:
            status = "✓" if result.success else "✗"
            execution_context += f"- {status} {result.task_id}: {result.result if result.success else result.error}\n"

    system_prompt = """You are a task planning agent. Create a detailed plan to accomplish the user's request.

You have access to MCP tools. When planning tasks, specify tool calls in format:
- tool_call: "server_name:tool_name"
- tool_input: dictionary of parameters for the tool

You can also include LLM reasoning tasks (no tool_call) for analysis or synthesis.

Return a JSON with:
{
  "reasoning": "explanation of your approach",
  "tasks": [
    {
      "id": "unique_task_id",
      "description": "what this task does",
      "tool_call": "server:tool" or null,
      "tool_input": {} or null,
      "depends_on": ["task_id_1", ...]  // tasks that must complete first
    }
  ]
}

Guidelines:
- Order tasks logically, respecting dependencies
- Break complex tasks into smaller steps
- Use available MCP tools when appropriate
- Include verification tasks if needed
"""

    user_prompt = f"""User Request: {user_message}

{execution_context if execution_context else ""}

Create a plan to accomplish this request."""

    try:
        # Use structured output to get the plan
        plan = llm.invoke_structured(user_prompt, Plan, system_message=system_prompt)

        # Ensure all tasks have IDs
        for i, task in enumerate(plan.tasks):
            if not task.id:
                task.id = _generate_task_id(i, task.description)

        # Update state
        new_state = state.copy()
        new_state["plan"] = plan
        new_state["current_task_index"] = 0
        new_state["error"] = None

        return new_state

    except Exception as e:
        new_state = state.copy()
        new_state["error"] = f"Planning failed: {str(e)}"
        return new_state
