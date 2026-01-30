"""LangGraph workflow definition for the hierarchical agent."""

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import architect, auditor, executor, refiner, router
from .state import AgentState

if TYPE_CHECKING:
    from agent.llm.base import BaseLLMProvider
    from agent.mcp.executor import MCPExecutor


# Conditional edge functions
def should_continue(state: AgentState) -> Literal["route_skill", "end"]:
    """Determine if there are more milestones or if we are finished.

    Args:
        state: Current agent state

    Returns:
        "route_skill" to continue, "end" to finish
    """
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    if not milestones:
        return "end"

    if current_idx >= len(milestones):
        return "end"

    return "route_skill"


def check_validation(state: AgentState) -> Literal["next_milestone", "retry"]:
    """Decide if the milestone was met or needs retry.

    Args:
        state: Current agent state

    Returns:
        "next_milestone" to advance, "retry" to try again
    """
    verification_status = state.get("last_verification_status", "")

    if verification_status == "passed":
        return "next_milestone"

    return "retry"


def next_milestone(state: AgentState) -> AgentState:
    """Advance to the next milestone.

    Args:
        state: Current agent state

    Returns:
        Updated state with incremented index and reset fields
    """
    return {
        **state,
        "current_idx": state.get("current_idx", 0) + 1,
        "tactical_plan": [],  # Reset tactical plan
        "active_skill_context": "",  # Reset skill context
        "last_verification_status": "",  # Reset validation status
        "retry_count": 0,  # Reset retry count
    }


class AgentGraph:
    """Production-ready agent graph with dependency injection."""

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        mcp_executor: "MCPExecutor",
    ):
        """Initialize the agent graph.

        Args:
            llm_provider: LLM provider for planning and validation
            mcp_executor: MCP executor for tool execution
        """
        self.llm_provider = llm_provider
        self.mcp_executor = mcp_executor
        self._compiled_graph = None

    def _create_node_functions(self):
        """Create node functions with dependencies injected."""

        def architect_node(state: AgentState) -> AgentState:
            return architect.node(state, self.llm_provider)

        def router_node(state: AgentState) -> AgentState:
            return router.node(state)

        def refiner_node(state: AgentState) -> AgentState:
            return refiner.node(state, self.llm_provider)

        def executor_node(state: AgentState) -> AgentState:
            return executor.node(state, self.mcp_executor)

        def auditor_node(state: AgentState) -> AgentState:
            return auditor.node(state, self.llm_provider)

        return {
            "architect": architect_node,
            "router": router_node,
            "refiner": refiner_node,
            "executor": executor_node,
            "auditor": auditor_node,
        }

    def compile(self):
        """Compile the agent graph.

        Returns:
            Compiled StateGraph ready for execution
        """
        if self._compiled_graph is not None:
            return self._compiled_graph

        workflow = StateGraph(AgentState)

        # Get node functions with injected dependencies
        nodes = self._create_node_functions()

        # Add nodes
        workflow.add_node("architect", nodes["architect"])
        workflow.add_node("router", nodes["router"])
        workflow.add_node("refiner", nodes["refiner"])
        workflow.add_node("executor", nodes["executor"])
        workflow.add_node("auditor", nodes["auditor"])
        workflow.add_node("next_milestone", next_milestone)

        # Define edges
        workflow.add_edge(START, "architect")

        # After planning, check if we have milestones to process
        workflow.add_conditional_edges("architect", should_continue, {"route_skill": "router", "end": END})

        # The Skill Loop: Routing -> Refining -> Executing
        workflow.add_edge("router", "refiner")
        workflow.add_edge("refiner", "executor")
        workflow.add_edge("executor", "auditor")

        # The Validation Loop (Self-Healing)
        workflow.add_conditional_edges(
            "auditor",
            check_validation,
            {
                "next_milestone": "next_milestone",
                "retry": "router",  # Try again with fresh context
            },
        )

        # After advancing milestone, check if we continue or end
        workflow.add_conditional_edges("next_milestone", should_continue, {"route_skill": "router", "end": END})

        self._compiled_graph = workflow.compile()
        return self._compiled_graph


def create_agent_graph(
    llm_provider: "BaseLLMProvider",
    mcp_executor: "MCPExecutor",
):
    """Create and compile the agent graph.

    Args:
        llm_provider: LLM provider for planning and validation
        mcp_executor: MCP executor for tool execution

    Returns:
        Compiled StateGraph ready for execution
    """
    graph = AgentGraph(llm_provider, mcp_executor)
    return graph.compile()
