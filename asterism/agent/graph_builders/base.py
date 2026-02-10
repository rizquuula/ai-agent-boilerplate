"""Base utilities for graph builders."""

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from asterism.agent.nodes.evaluator.router import RouteTarget, determine_route
from asterism.agent.state import AgentState

if TYPE_CHECKING:
    from asterism.agent.agent import Agent


def add_common_nodes(workflow: StateGraph, agent: "Agent") -> None:
    """Add planner, executor, and evaluator nodes to the workflow.

    Args:
        workflow: The StateGraph to add nodes to.
        agent: The Agent instance with dependencies.
    """
    workflow.add_node("planner_node", _make_planner_node(agent))
    workflow.add_node("executor_node", _make_executor_node(agent))
    workflow.add_node("evaluator_node", _make_evaluator_node(agent))


def add_common_edges(workflow: StateGraph) -> None:
    """Add edges from START→planner→executor→evaluator.

    Args:
        workflow: The StateGraph to add edges to.
    """
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_edge("executor_node", "evaluator_node")


def make_routing_function(agent: "Agent"):
    """Create standard routing function for evaluator.

    Args:
        agent: The Agent instance (for consistency, though not used directly).

    Returns:
        Routing function that returns RouteTarget values.
    """

    def _route(state: AgentState) -> str:
        route = determine_route(state)
        return str(route)

    return _route


def make_routing_function_with_end(agent: "Agent"):
    """Create routing function that routes FINALIZER to END.

    Use this for streaming graph where we want to stop before finalization.

    Args:
        agent: The Agent instance (for consistency, though not used directly).

    Returns:
        Routing function that returns END when evaluation decides to finalize.
    """

    def _route(state: AgentState) -> str:
        route = determine_route(state)
        if route == RouteTarget.FINALIZER:
            return END
        return str(route)

    return _route


def _make_planner_node(agent: "Agent"):
    """Create planner node with dependencies injected."""
    from asterism.agent.nodes import planner_node

    llm = agent.llm
    mcp_executor = agent.mcp_executor
    workspace_root = agent.workspace_root

    def _node(state: AgentState) -> AgentState:
        return planner_node(llm, mcp_executor, state, workspace_root)

    return _node


def _make_executor_node(agent: "Agent"):
    """Create executor node with dependencies injected."""
    from asterism.agent.nodes import executor_node

    llm = agent.llm
    mcp_executor = agent.mcp_executor

    def _node(state: AgentState) -> AgentState:
        return executor_node(llm, mcp_executor, state)

    return _node


def _make_evaluator_node(agent: "Agent"):
    """Create evaluator node with dependencies injected."""
    from asterism.agent.nodes import evaluator_node

    llm = agent.llm

    def _node(state: AgentState) -> AgentState:
        return evaluator_node(llm, state)

    return _node


def _make_finalizer_node(agent: "Agent"):
    """Create finalizer node with dependencies injected."""
    from asterism.agent.nodes import finalizer_node

    llm = agent.llm

    def _node(state: AgentState) -> AgentState:
        return finalizer_node(llm, state)

    return _node
