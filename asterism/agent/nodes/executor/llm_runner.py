"""LLM task execution runner."""

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import TaskResult
from asterism.agent.nodes.shared import LLMCaller
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider

# Simple system prompt for LLM tasks
LLM_TASK_SYSTEM_PROMPT = """You are a helpful assistant executing a specific task.
Follow the instructions carefully and provide a clear, concise response."""


class LLMRunner:
    """Runner for LLM-only execution tasks."""

    def __init__(self, llm: BaseLLMProvider):
        self.caller = LLMCaller(llm, "executor_node")

    def execute(self, task, state: AgentState) -> TaskResult:
        """Execute an LLM-only task.

        Args:
            task: Task with description to process.
            state: Current agent state for context.

        Returns:
            TaskResult with execution outcome.
        """
        if not task.description:
            return TaskResult(
                task_id=task.id,
                success=False,
                result=None,
                error="Task description is required for LLM-only tasks",
            )

        context = self._build_context(task, state)
        messages = self._build_messages(task.description, context)

        try:
            result = self.caller.call_text(messages, f"executing LLM task {task.id}")

            return TaskResult(
                task_id=task.id,
                success=True,
                result=result.parsed,
                llm_usage=result.usage,
            )

        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                result=None,
                error=str(e),
            )

    def _build_context(self, task, state: AgentState) -> str:
        """Build execution context from dependent task results.

        Args:
            task: The task being executed.
            state: Current agent state.

        Returns:
            Formatted context string from dependencies.
        """
        if not task.depends_on:
            return ""

        execution_results = state.get("execution_results", [])
        dependent_results = [r for r in execution_results if r.task_id in task.depends_on]

        if not dependent_results:
            return ""

        lines = ["\n\nContext from previous tasks:"]

        for dep_result in dependent_results:
            lines.append(f"\n--- Result from task '{dep_result.task_id}' ---")
            if dep_result.success:
                lines.append(str(dep_result.result))
            else:
                lines.append(f"Task failed: {dep_result.error}")

        return "\n".join(lines)

    def _build_messages(self, description: str, context: str) -> list:
        """Build LLM messages for the task.

        Args:
            description: Task description.
            context: Context from dependencies.

        Returns:
            List of LangChain messages.
        """
        full_prompt = f"{description}\n\n{context}" if context else description

        return [
            SystemMessage(content=LLM_TASK_SYSTEM_PROMPT),
            HumanMessage(content=full_prompt),
        ]
