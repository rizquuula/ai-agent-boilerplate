"""Unit tests for agent nodes."""

from unittest.mock import MagicMock

from asterism.agent.models import EvaluationDecision, EvaluationResult, Plan, Task, TaskResult
from asterism.agent.nodes import evaluator_node, executor_node, finalizer_node, planner_node, should_continue
from asterism.agent.state import AgentState


class TestPlannerNode:
    """Tests for planner_node."""

    def test_planner_node_creates_plan(self, mock_llm, mock_mcp_executor):
        """Test that planner_node creates a plan."""
        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": None,
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = planner_node(mock_llm, mock_mcp_executor, state)

        assert result["plan"] is not None
        assert isinstance(result["plan"], Plan)
        assert result["plan"].reasoning
        assert len(result["plan"].tasks) > 0
        assert result["current_task_index"] == 0
        assert result["error"] is None

    def test_planner_node_handles_error(self, mock_llm, mock_mcp_executor):
        """Test that planner_node handles errors gracefully."""
        mock_llm.invoke_structured.side_effect = Exception("LLM error")

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": None,
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = planner_node(mock_llm, mock_mcp_executor, state)

        assert result["error"] is not None
        assert "Planning failed" in result["error"]


class TestExecutorNode:
    """Tests for executor_node."""

    def test_executor_node_with_tool_call(self, mock_llm, mock_mcp_executor):
        """Test executor_node executing a tool call."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(
                tasks=[
                    Task(
                        id="task_1",
                        description="Test task",
                        tool_call="test:tool",
                        tool_input={"param": "value"},
                        depends_on=[],
                    )
                ],
                reasoning="Test reasoning",
            ),
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = executor_node(mock_llm, mock_mcp_executor, state)

        assert len(result["execution_results"]) == 1
        task_result = result["execution_results"][0]
        assert task_result.success is True
        assert task_result.task_id == "task_1"
        assert result["current_task_index"] == 1
        assert result["error"] is None

    def test_executor_node_with_llm_task(self, mock_llm, mock_mcp_executor):
        """Test executor_node executing an LLM-only task."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(
                tasks=[
                    Task(
                        id="task_1",
                        description="Analyze this",
                        tool_call=None,
                        tool_input=None,
                        depends_on=[],
                    )
                ],
                reasoning="Test reasoning",
            ),
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = executor_node(mock_llm, mock_mcp_executor, state)

        assert len(result["execution_results"]) == 1
        task_result = result["execution_results"][0]
        assert task_result.success is True
        assert task_result.result == "Mock LLM response"

    def test_executor_node_handles_tool_failure(self, mock_llm, mock_mcp_executor):
        """Test executor_node handling tool failure."""
        mock_mcp_executor.execute_tool.return_value = {
            "success": False,
            "error": "Tool failed",
            "tool": "test:tool",
            "tool_call": "test:tool",
        }

        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(
                tasks=[
                    Task(
                        id="task_1",
                        description="Test task",
                        tool_call="test:tool",
                        tool_input={},
                        depends_on=[],
                    )
                ],
                reasoning="Test reasoning",
            ),
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = executor_node(mock_llm, mock_mcp_executor, state)

        assert len(result["execution_results"]) == 1
        task_result = result["execution_results"][0]
        assert task_result.success is False
        assert task_result.error == "Tool failed"
        assert result["error"] == "Tool failed"

    def test_executor_node_no_plan(self, mock_llm, mock_mcp_executor):
        """Test executor_node with no plan."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": None,
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = executor_node(mock_llm, mock_mcp_executor, state)

        assert result["error"] == "No plan to execute"

    def test_executor_node_all_complete(self, mock_llm, mock_mcp_executor):
        """Test executor_node when all tasks are complete."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(tasks=[], reasoning="Test"),
            "current_task_index": 1,  # Beyond end
            "execution_results": [],
            "final_response": None,
            "error": None,
        }

        result = executor_node(mock_llm, mock_mcp_executor, state)

        assert result["error"] == "All tasks already completed"


class TestEvaluatorNode:
    """Tests for evaluator_node."""

    def test_evaluator_node_with_llm_decision(self, mock_llm):
        """Test evaluator_node uses LLM to make decision."""
        def mock_evaluator(prompt, schema, **kwargs):
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                reasoning="Execution on track",
            )
        
        mock_llm.invoke_structured.side_effect = mock_evaluator

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=True, result="ok", error=None)],
            "final_response": None,
            "error": None,
        }

        result = evaluator_node(mock_llm, state)

        assert isinstance(result, dict)
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].decision == EvaluationDecision.CONTINUE
        assert result["evaluation_result"].reasoning == "Execution on track"
        mock_llm.invoke_structured.assert_called_once()

    def test_evaluator_node_replan_decision(self, mock_llm):
        """Test evaluator_node triggers replanning when LLM decides to replan."""
        def mock_evaluator(prompt, schema, **kwargs):
            return EvaluationResult(
                decision=EvaluationDecision.REPLAN,
                reasoning="Unexpected result requires new approach",
                suggested_changes="Use different tool",
            )
        
        mock_llm.invoke_structured.side_effect = mock_evaluator

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=False, result=None, error="failed")],
            "final_response": None,
            "error": None,
        }

        result = evaluator_node(mock_llm, state)

        assert isinstance(result, dict)
        assert result["evaluation_result"].decision == EvaluationDecision.REPLAN
        assert result["error"] is not None
        assert "Replanning needed" in result["error"]
        # Check that replanning context was added to messages
        assert len(result["messages"]) > len(state["messages"])

    def test_evaluator_node_finalize_decision(self, mock_llm):
        """Test evaluator_node handles finalize decision."""
        def mock_evaluator(prompt, schema, **kwargs):
            return EvaluationResult(
                decision=EvaluationDecision.FINALIZE,
                reasoning="Goals achieved early",
            )
        
        mock_llm.invoke_structured.side_effect = mock_evaluator

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": Plan(
                tasks=[
                    Task(id="t1", description="t1", tool_call=None),
                    Task(id="t2", description="t2", tool_call=None),
                ],
                reasoning="r",
            ),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=True, result="complete data", error=None)],
            "final_response": None,
            "error": None,
        }

        result = evaluator_node(mock_llm, state)

        assert isinstance(result, dict)
        assert result["evaluation_result"].decision == EvaluationDecision.FINALIZE
        assert result["error"] is None  # No error for finalize

    def test_evaluator_node_fallback_on_llm_error(self, mock_llm):
        """Test evaluator_node falls back to logic when LLM fails."""
        mock_llm.invoke_structured.side_effect = Exception("LLM error")

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Test request")],
            "plan": Plan(
                tasks=[
                    Task(id="t1", description="t1", tool_call=None),
                    Task(id="t2", description="t2", tool_call=None),
                ],
                reasoning="r",
            ),
            "current_task_index": 1,  # One task done, one remaining
            "execution_results": [TaskResult(task_id="t1", success=True, result="ok", error=None)],
            "final_response": None,
            "error": None,
        }

        result = evaluator_node(mock_llm, state)

        assert isinstance(result, dict)
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].decision == EvaluationDecision.CONTINUE  # Fallback continues since more tasks remain
        assert "LLM evaluation failed" in result["evaluation_result"].reasoning

    def test_evaluator_node_finalize_complete(self):
        """Test should_continue signals finalizer when all tasks complete."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=True, result="ok", error=None)],
            "final_response": None,
            "error": None,
        }

        route = should_continue(state)
        assert route == "finalizer_node"

    def test_evaluator_node_route_to_executor(self):
        """Test should_continue routes to executor when more tasks."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(
                tasks=[
                    Task(id="t1", description="t1", tool_call=None),
                    Task(id="t2", description="t2", tool_call=None),
                ],
                reasoning="r",
            ),
            "current_task_index": 1,  # On second task
            "execution_results": [TaskResult(task_id="t1", success=True, result="ok", error=None)],
            "final_response": None,
            "error": None,
        }

        route = should_continue(state)
        assert route == "executor_node"

    def test_evaluator_node_route_to_planner_on_error(self):
        """Test should_continue routes to planner when error present."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(tasks=[], reasoning="r"),
            "current_task_index": 0,
            "execution_results": [],
            "final_response": None,
            "error": "Something went wrong",
        }

        route = should_continue(state)
        assert route == "planner_node"

    def test_should_continue_uses_evaluation_result(self):
        """Test should_continue uses LLM evaluation result when available."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [],
            "final_response": None,
            "error": None,
            "evaluation_result": EvaluationResult(
                decision=EvaluationDecision.FINALIZE,
                reasoning="Early completion",
            ),
        }

        route = should_continue(state)
        assert route == "finalizer_node"

    def test_should_continue_replan_from_evaluation(self):
        """Test should_continue routes to planner when evaluation says replan."""
        state: AgentState = {
            "session_id": "test",
            "messages": [],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [],
            "final_response": None,
            "error": None,
            "evaluation_result": EvaluationResult(
                decision=EvaluationDecision.REPLAN,
                reasoning="Need different approach",
            ),
        }

        route = should_continue(state)
        assert route == "planner_node"


class TestFinalizerNode:
    """Tests for finalizer_node."""

    def test_finalizer_node_success(self, mock_llm):
        """Test finalizer_node generates response on success."""
        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Original request")],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=True, result="Result data", error=None)],
            "final_response": None,
            "error": None,
        }

        result = finalizer_node(mock_llm, state)

        assert result["final_response"] is not None
        assert result["final_response"].message
        assert len(result["final_response"].execution_trace) == 1
        assert result["final_response"].plan_used == state["plan"]
        assert result["error"] is None

    def test_finalizer_node_with_failures(self, mock_llm):
        """Test finalizer_node handles failed tasks."""
        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Original request")],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=False, result=None, error="Task error")],
            "final_response": None,
            "error": None,
        }

        result = finalizer_node(mock_llm, state)

        assert result["final_response"] is not None
        assert "error" in result["final_response"].message.lower()
        assert len(result["final_response"].execution_trace) == 1

    def test_finalizer_node_llm_failure(self, mock_llm):
        """Test finalizer_node fallback when LLM fails."""
        mock_llm.invoke.side_effect = Exception("LLM error")

        state: AgentState = {
            "session_id": "test",
            "messages": [MagicMock(content="Original request")],
            "plan": Plan(tasks=[Task(id="t1", description="t", tool_call=None)], reasoning="r"),
            "current_task_index": 1,
            "execution_results": [TaskResult(task_id="t1", success=True, result="Result", error=None)],
            "final_response": None,
            "error": None,
        }

        result = finalizer_node(mock_llm, state)

        assert result["final_response"] is not None
        assert "failed" in result["final_response"].message.lower()
