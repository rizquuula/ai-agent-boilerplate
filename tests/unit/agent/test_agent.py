"""Test main Agent class."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from asterism.agent.agent import Agent, _initialize_state
from asterism.agent.models import AgentResponse, LLMUsage, Plan, Task
from asterism.agent.state import AgentState


def create_test_messages(content: str = "Hello, agent!") -> list[BaseMessage]:
    """Create a test message list."""
    return [HumanMessage(content=content)]


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.name = "test_llm"
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_mcp_executor():
    """Create a mock MCP executor."""
    executor = MagicMock()
    executor.get_available_tools.return_value = {"test_server": ["tool1", "tool2"]}
    return executor


def test_initialize_state():
    """Test state initialization."""
    messages = create_test_messages("Hello, agent!")
    state = _initialize_state("session_123", messages)

    assert state["session_id"] == "session_123"
    assert state["trace_id"] is not None
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "Hello, agent!"
    assert state["plan"] is None
    assert state["current_task_index"] == 0
    assert state["execution_results"] == []
    assert state["final_response"] is None
    assert state["error"] is None
    assert state["llm_usage"] == []


def test_initialize_state_with_multiple_messages():
    """Test state initialization with system and user messages."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
    ]
    state = _initialize_state("session_123", messages)

    assert state["session_id"] == "session_123"
    assert len(state["messages"]) == 2
    assert isinstance(state["messages"][0], SystemMessage)
    assert isinstance(state["messages"][1], HumanMessage)


def test_agent_initialization_defaults(mock_llm, mock_mcp_executor):
    """Test Agent initialization with default values."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

    assert agent.llm is mock_llm
    assert agent.mcp_executor is mock_mcp_executor
    assert agent.db_path == ".checkpoints/agent.db"
    assert agent.workspace_root == "./workspace"
    assert agent._graph is None
    assert agent._checkpointer is None
    assert agent._conn is None


def test_agent_initialization_custom(mock_llm, mock_mcp_executor):
    """Test Agent initialization with custom values."""
    agent = Agent(
        llm=mock_llm,
        mcp_executor=mock_mcp_executor,
        db_path="/custom/path.db",
        workspace_root="/custom/workspace",
    )

    assert agent.db_path == "/custom/path.db"
    assert agent.workspace_root == "/custom/workspace"


def test_agent_initialization_none_db_path(mock_llm, mock_mcp_executor):
    """Test Agent initialization with None db_path uses default."""
    agent = Agent(
        llm=mock_llm,
        mcp_executor=mock_mcp_executor,
        db_path=None,
    )

    assert agent.db_path == ".checkpoints/agent.db"


@patch("asterism.agent.agent.SqliteSaver")
@patch("sqlite3.connect")
def test_get_checkpointer(mock_sqlite_connect, mock_sqlite_saver, mock_llm, mock_mcp_executor):
    """Test getting checkpointer creates it on demand."""
    mock_conn = MagicMock()
    mock_sqlite_connect.return_value = mock_conn
    mock_saver_instance = MagicMock()
    mock_sqlite_saver.return_value = mock_saver_instance

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor, db_path=":memory:")

    # First call creates the checkpointer
    checkpointer = agent._get_checkpointer()

    assert checkpointer is mock_saver_instance
    assert agent._checkpointer is mock_saver_instance
    assert agent._conn is mock_conn
    mock_sqlite_connect.assert_called_once_with(":memory:", check_same_thread=False)


@patch("asterism.agent.agent.SqliteSaver")
@patch("sqlite3.connect")
def test_get_checkpointer_caches_result(mock_sqlite_connect, mock_sqlite_saver, mock_llm, mock_mcp_executor):
    """Test that checkpointer is cached."""
    mock_conn = MagicMock()
    mock_sqlite_connect.return_value = mock_conn
    mock_saver_instance = MagicMock()
    mock_sqlite_saver.return_value = mock_saver_instance

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor, db_path="test.db")

    # Multiple calls should return same instance
    cp1 = agent._get_checkpointer()
    cp2 = agent._get_checkpointer()

    assert cp1 is cp2
    mock_sqlite_connect.assert_called_once()


@patch("asterism.agent.agent.StateGraph")
def test_agent_build_creates_graph(mock_state_graph_class, mock_llm, mock_mcp_executor):
    """Test that build() creates the workflow graph."""
    mock_workflow = MagicMock()
    mock_state_graph_class.return_value = mock_workflow
    mock_compiled = MagicMock()
    mock_workflow.compile.return_value = mock_compiled

    with patch.object(Agent, "_get_checkpointer") as mock_get_cp:
        mock_get_cp.return_value = MagicMock()

        agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
        result = agent.build()

        assert result is mock_compiled
        # Verify nodes were added (check call count and names)
        add_node_calls = [call for call in mock_workflow.add_node.call_args_list]
        node_names = [call[0][0] for call in add_node_calls]
        assert "planner_node" in node_names
        assert "executor_node" in node_names
        assert "evaluator_node" in node_names
        assert "finalizer_node" in node_names


@patch("asterism.agent.agent.StateGraph")
def test_agent_build_caches_graph(mock_state_graph_class, mock_llm, mock_mcp_executor):
    """Test that build() caches the graph."""
    mock_workflow = MagicMock()
    mock_state_graph_class.return_value = mock_workflow
    mock_compiled = MagicMock()
    mock_workflow.compile.return_value = mock_compiled

    with patch.object(Agent, "_get_checkpointer") as mock_get_cp:
        mock_get_cp.return_value = MagicMock()

        agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

        # Build twice
        result1 = agent.build()
        result2 = agent.build()

        assert result1 is result2
        # StateGraph should only be instantiated once
        mock_state_graph_class.assert_called_once()


def test_make_planner_node(mock_llm, mock_mcp_executor):
    """Test creating planner node function."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor, workspace_root="/workspace")

    planner_fn = agent._make_planner_node()

    assert callable(planner_fn)


def test_make_executor_node(mock_llm, mock_mcp_executor):
    """Test creating executor node function."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

    executor_fn = agent._make_executor_node()

    assert callable(executor_fn)


def test_make_evaluator_node(mock_llm, mock_mcp_executor):
    """Test creating evaluator node function."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

    evaluator_fn = agent._make_evaluator_node()

    assert callable(evaluator_fn)


def test_make_finalizer_node(mock_llm, mock_mcp_executor):
    """Test creating finalizer node function."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

    finalizer_fn = agent._make_finalizer_node()

    assert callable(finalizer_fn)


def test_make_routing_function(mock_llm, mock_mcp_executor):
    """Test creating routing function."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)

    route_fn = agent._make_routing_function()

    assert callable(route_fn)


@patch.object(Agent, "build")
def test_invoke_successful_execution(mock_build, mock_llm, mock_mcp_executor):
    """Test successful invocation."""
    mock_graph = MagicMock()
    mock_build.return_value = mock_graph

    # Create mock plan
    plan_used = Plan(tasks=[Task(id="t1", description="Test task")], reasoning="Test")

    # Create mock final state
    final_response = AgentResponse(
        message="Task completed successfully!",
        execution_trace=[{"task_id": "t1", "success": True}],
        plan_used=plan_used,
    )
    final_state: AgentState = {
        "session_id": "session_123",
        "trace_id": "trace_456",
        "messages": [],
        "plan": plan_used,
        "current_task_index": 1,
        "execution_results": [],
        "evaluation_result": None,
        "final_response": final_response,
        "error": None,
        "llm_usage": [
            LLMUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                model="gpt-4",
                node_name="planner_node",
            )
        ],
    }
    mock_graph.invoke.return_value = final_state

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    messages = create_test_messages("Do something")
    result = agent.invoke("session_123", messages)

    assert result["message"] == "Task completed successfully!"
    assert result["session_id"] == "session_123"
    assert len(result["execution_trace"]) == 1
    assert result["plan_used"] is not None
    assert "total_usage" in result
    assert result["total_usage"]["total_tokens"] == 150


@patch.object(Agent, "build")
def test_invoke_graph_execution_error(mock_build, mock_llm, mock_mcp_executor):
    """Test invocation when graph execution fails."""
    mock_graph = MagicMock()
    mock_build.return_value = mock_graph
    mock_graph.invoke.side_effect = Exception("Graph execution failed")

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    messages = create_test_messages("Do something")
    result = agent.invoke("session_123", messages)

    assert "error" in result
    assert "Graph execution failed" in result["message"]
    assert result["execution_trace"] == []
    assert result["plan_used"] is None


@patch.object(Agent, "build")
def test_invoke_no_final_response(mock_build, mock_llm, mock_mcp_executor):
    """Test invocation when no final response is generated."""
    mock_graph = MagicMock()
    mock_build.return_value = mock_graph

    final_state: AgentState = {
        "session_id": "session_123",
        "trace_id": "trace_456",
        "messages": [],
        "plan": None,
        "current_task_index": 0,
        "execution_results": [],
        "evaluation_result": None,
        "final_response": None,
        "error": None,
        "llm_usage": [],
    }
    mock_graph.invoke.return_value = final_state

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    messages = create_test_messages("Do something")
    result = agent.invoke("session_123", messages)

    assert "error" in result
    assert "did not produce a response" in result["message"]


@patch.object(Agent, "build")
def test_invoke_usage_aggregation(mock_build, mock_llm, mock_mcp_executor):
    """Test that usage is properly aggregated across nodes."""
    mock_graph = MagicMock()
    mock_build.return_value = mock_graph

    final_response = AgentResponse(
        message="Done!",
        execution_trace=[],
    )
    final_state: AgentState = {
        "session_id": "session_123",
        "trace_id": "trace_456",
        "messages": [],
        "plan": None,
        "current_task_index": 0,
        "execution_results": [],
        "evaluation_result": None,
        "final_response": final_response,
        "error": None,
        "llm_usage": [
            LLMUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150, model="gpt-4", node_name="planner_node"
            ),
            LLMUsage(
                prompt_tokens=200, completion_tokens=100, total_tokens=300, model="gpt-4", node_name="executor_node"
            ),
            LLMUsage(
                prompt_tokens=50, completion_tokens=25, total_tokens=75, model="gpt-4", node_name="planner_node"
            ),  # Second planner call
        ],
    }
    mock_graph.invoke.return_value = final_state

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    messages = create_test_messages("Do something")
    result = agent.invoke("session_123", messages)

    usage = result["total_usage"]
    assert usage["total_prompt_tokens"] == 350
    assert usage["total_completion_tokens"] == 175
    assert usage["total_tokens"] == 525
    assert usage["calls_by_node"]["planner_node"] == 2
    assert usage["calls_by_node"]["executor_node"] == 1


@patch.object(Agent, "build")
def test_clear_session(mock_build, mock_llm, mock_mcp_executor):
    """Test clearing a session."""
    mock_checkpointer = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_checkpointer.conn = mock_conn

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    agent._checkpointer = mock_checkpointer

    agent.clear_session("session_123")

    mock_cursor.execute.assert_called_once_with(
        "DELETE FROM checkpoints WHERE thread_id = ?",
        ("session_123",),
    )
    mock_conn.commit.assert_called_once()


@patch.object(Agent, "build")
def test_clear_session_builds_first(mock_build, mock_llm, mock_mcp_executor):
    """Test that clear_session builds the checkpointer if needed."""
    mock_checkpointer = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_checkpointer.conn = mock_conn

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    agent._checkpointer = None

    # Mock build to set checkpointer
    def mock_build_impl():
        agent._checkpointer = mock_checkpointer
        return MagicMock()

    mock_build.side_effect = mock_build_impl

    agent.clear_session("session_123")

    mock_build.assert_called_once()


def test_close_with_connection(mock_llm, mock_mcp_executor):
    """Test closing database connection."""
    mock_conn = MagicMock()

    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    agent._conn = mock_conn

    agent.close()

    mock_conn.close.assert_called_once()
    # Note: The actual implementation doesn't set _conn to None after closing


def test_close_without_connection(mock_llm, mock_mcp_executor):
    """Test closing when no connection exists."""
    agent = Agent(llm=mock_llm, mcp_executor=mock_mcp_executor)
    agent._conn = None

    # Should not raise any errors
    agent.close()
