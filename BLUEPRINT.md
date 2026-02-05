# Asterism Agent - Project Blueprint

## 1. Overview

Asterism is a high-performance, hierarchical AI agent framework built with **LangGraph** for workflow orchestration and **MCP (Model Context Protocol)** for environmental interaction. The framework implements a "Plan-Execute-Evaluate" cycle with automatic validation and retry logic, enabling robust autonomous task completion.

**Key Characteristics:**
- Multi-level task decomposition from objectives to atomic actions
- Dynamic LLM provider abstraction (OpenAI, with extensibility for others)
- Configuration-driven MCP tool execution with multiple transport protocols
- State persistence via SQLite checkpointing
- Comprehensive error handling and self-healing workflows
- Full test coverage with unit and integration tests

**Primary Use Case:** Building autonomous agents that can plan, execute tools, validate results, and recover from failures in a structured manner.

---

## 2. Directory Structure

```
asterism/                    # Main package
├── __init__.py
├── main.py                  # Application entry point (currently empty placeholder)
│
├── agent/                   # Core agent implementation
│   ├── __init__.py
│   ├── agent.py            # Agent class with LangGraph workflow
│   ├── state/
│   │   ├── __init__.py
│   │   └── agent_state.py  # AgentState TypedDict definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── planner.py      # Task planning node
│   │   ├── executor.py     # Task execution node
│   │   ├── evaluator.py    # Evaluation & routing node
│   │   └── finalizer.py    # Final response generation node
│   └── models/
│       ├── __init__.py
│       └── schemas.py      # Pydantic models (Task, Plan, TaskResult, AgentResponse)
│
├── llm/                     # LLM provider implementations
│   ├── __init__.py
│   ├── base.py             # Abstract BaseLLMProvider interface
│   └── openai_provider.py  # OpenAI implementation
│
└── mcp/                     # MCP (Model Context Protocol) integration
    ├── __init__.py
    ├── config.py           # MCP server configuration management
    ├── executor.py         # Dynamic MCP tool executor
    └── transport_executor/
        ├── __init__.py
        ├── base.py         # Abstract BaseTransport interface
        ├── stdio.py        # STDIO transport implementation
        ├── http_stream.py  # HTTP stream transport
        └── sse.py          # Server-Sent Events transport

tests/                       # Test suite
├── unit/                    # Unit tests
│   ├── agent/
│   │   ├── test_agent.py
│   │   └── test_nodes.py
│   ├── mcp/
│   │   ├── test_config.py
│   │   ├── test_executor.py
│   │   └── transport_executor/
│   │       ├── test_base.py
│   │       ├── test_factory.py
│   │       ├── test_http_stream.py
│   │       ├── test_sse.py
│   │       └── test_stdio.py
│   └── llm/ (integration)
│       └── test_openai_provider.py
│
└── integration_tests/      # Integration tests
    ├── conftest.py
    ├── llm/
    │   └── test_openai_provider.py
    └── transport_executor/
        ├── test_stdio.py
        ├── test_http_stream.py
        └── test_sse.py

workspace/                   # Development workspace (not part of package)
├── AGENT.md                 # Placeholder for agent documentation
├── SOUL.md                  # Placeholder for project philosophy
├── TOOLS.md                 # Placeholder for tools documentation
├── mcp_servers/            # Example MCP server implementations
│   ├── mcp_servers.json    # MCP server configuration
│   └── localtime_mcp/      # Sample MCP server (time service)
└── skills/                 # Skill definitions (persona configurations)
    ├── code-analyzer/
    ├── code-reader/
    ├── filesystem/
    └── report-writer/

config/                     # Configuration directory (if exists)
├── mcp_servers.json        # MCP server definitions (expected location)

.env.example                # Environment variable template
pyproject.toml              # Project dependencies and tool configs
uv.lock                     # Locked dependency versions
auto_format_ruff.sh         # Code formatting script
```

---

## 3. Architecture

### 3.1 High-Level Workflow

The agent operates as a **cyclic state machine** using LangGraph for orchestration:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Workflow Cycle                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │   START     │                                           │
│  └──────┬──────┘                                           │
│         ▼                                                  │
│  ┌─────────────────┐   Plan creation/update                │
│  │  Planner Node   │◄────────────────┐                    │
│  └────────┬────────┘                │                    │
│           │                          │                    │
│           ▼                          │                    │
│  ┌─────────────────┐   Execute task │                    │
│  │ Executor Node   │◄───────────────┘                    │
│  └────────┬────────┘                                      │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────┐   Evaluate result                    │
│  │ Evaluator Node  │◄────────────────┐                    │
│  └────────┬────────┘                │                    │
│           │                          │                    │
│           ▼                          │                    │
│  ┌─────────────────┐   All tasks done?                    │
│  │ Should_Continue │───────────────────────┐              │
│  └────────┬────────┘                       │              │
│           │                                 │              │
│           ▼                                 │              │
│    ┌──────┴──────┐                          │              │
│    │             │                          │              │
│    ▼             ▼                          ▼              │
│ ┌───────┐   ┌─────────┐            ┌─────────────┐      │
│ │ Re-   │   │ Next    │            │  Finalizer  │      │
│ │ plan  │   │ task    │            │    Node     │      │
│ └───────┘   └─────────┘            └─────────────┘      │
│     │             │                       │              │
│     └─────────────┴───────────────────────┘              │
│                      Loop back                             │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │                  END (Final State)           │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### Agent Class (`asterism/agent/agent.py`)
- **Purpose:** Main entry point for the agent system
- **Responsibilities:**
  - Build and compile the LangGraph state machine
  - Manage SQLite checkpoint storage for state persistence
  - Provide `invoke()` method for processing user requests
  - Handle session lifecycle (clear_session, close)
- **Dependencies:** LLM provider, MCP executor, checkpoint storage

#### State Management (`asterism/agent/state/agent_state.py`)
- **Type:** `TypedDict` for LangGraph compatibility
- **Fields:**
  - `session_id`: Unique session identifier
  - `messages`: List of LangChain message objects
  - `plan`: Current execution plan (Plan model)
  - `current_task_index`: Index of active task
  - `execution_results`: List of TaskResult objects
  - `final_response`: Final AgentResponse (populated at end)
  - `error`: Error message if any

#### Nodes (Workflow Steps)

1. **Planner Node** (`nodes/planner.py`)
   - Creates or updates a structured plan using LLM
   - Outputs: `Plan` with ordered `Task` objects
   - Each task specifies: description, optional tool_call (server:tool), tool_input, dependencies

2. **Executor Node** (`nodes/executor.py`)
   - Executes the current task based on its type:
     - **Tool task:** Calls MCP executor to run `server_name:tool_name`
     - **LLM task:** Directly invokes LLM with task description
   - Validates dependencies before execution
   - Produces `TaskResult` (success/failure, result/error)

3. **Evaluator Node** (`nodes/evaluator.py`)
   - Lightweight node that doesn't modify state
   - Triggers routing function `should_continue()` to determine next step

4. **Finalizer Node** (`nodes/finalizer.py`)
   - Synthesizes final natural language response using LLM
   - Packages execution trace and plan used
   - Handles both success and failure scenarios

#### LLM Provider Abstraction (`asterism/llm/`)
- **Base Interface:** `BaseLLMProvider` (abstract)
  - `invoke(prompt, **kwargs) -> str`
  - `invoke_structured(prompt, schema, **kwargs) -> Any`
  - Properties: `name`, `model`
- **OpenAI Implementation:** `OpenAIProvider`
  - Uses LangChain's `ChatOpenAI`
  - Supports structured output via PydanticOutputParser
  - Configurable base_url for compatible endpoints

#### MCP Integration (`asterism/mcp/`)

**Configuration (`config.py`):**
- Loads server definitions from JSON file (default: `config/mcp_servers.json`)
- Manages server metadata: command, args, transport type, enabled status
- Provides methods: `get_enabled_servers()`, `is_server_enabled()`, `get_server_metadata()`

**Executor (`executor.py`):**
- Dynamic tool routing based on configuration
- Lazy transport initialization per server
- Tool caching (list_tools on first access)
- Methods: `execute_tool()`, `get_available_tools()`, `validate_tool_call()`, `shutdown()`
- Global singleton pattern with `get_mcp_executor()`

**Transport Layer (`transport_executor/`):**
- Abstract `BaseTransport` interface:
  - `start(command, args)`
  - `stop()`
  - `execute_tool(tool_name, **kwargs)`
  - `list_tools()`
  - `is_alive()`
- Implementations:
  - `stdio.py`: Subprocess with stdin/stdout (most common)
  - `http_stream.py`: HTTP with streaming responses
  - `sse.py`: Server-Sent Events

---

## 4. Data Models

All data models are defined in [`asterism/agent/models/schemas.py`](asterism/agent/models/schemas.py:1) using Pydantic for validation and serialization.

### Task
```python
class Task(BaseModel):
    id: str                                    # Unique identifier
    description: str                           # Human-readable description
    tool_call: str | None                     # Format: "server_name:tool_name"
    tool_input: dict[str, Any] | None         # Parameters for the tool
    depends_on: list[str]                     # Task IDs that must complete first
```

### Plan
```python
class Plan(BaseModel):
    tasks: list[Task]                         # Ordered task list
    reasoning: str                            # Explanation of approach
```

### TaskResult
```python
class TaskResult(BaseModel):
    task_id: str
    success: bool
    result: Any | None
    error: str | None
    timestamp: datetime                       # Auto-generated on creation
```

### AgentResponse
```python
class AgentResponse(BaseModel):
    message: str                              # Final natural language response
    execution_trace: list[dict[str, Any]]     # Full execution history
    plan_used: Plan | None                    # The plan that was executed
```

---

## 5. Matching Rules

### 5.1 Task Dependency Resolution
- Before executing a task, the executor checks that all `depends_on` task IDs have completed successfully
- If dependencies are unsatisfied, the task is skipped with an error, triggering replanning
- Dependencies form a directed acyclic graph (DAG); cycles are not explicitly detected but will cause infinite replanning

### 5.2 Routing Logic (`should_continue` in [`evaluator.py`](asterism/agent/nodes/evaluator.py:23))
The evaluator routes to:
- **`planner_node`** if:
  - An error exists in state
  - No plan exists
  - Last task failed
- **`executor_node`** if:
  - Plan exists, no errors, and `current_task_index < len(tasks)`
- **`finalizer_node`** if:
  - All tasks completed (`current_task_index >= len(tasks)`)

### 5.3 Tool Validation
- MCP executor validates:
  - Server is enabled in configuration (`is_server_enabled()`)
  - Tool name exists in server's tool list (cached via `list_tools()`)
- Invalid tool calls return `success=False` with descriptive error

### 5.4 LLM Structured Output
- Planner uses `llm.invoke_structured(prompt, Plan, system_message=...)`
- OpenAI provider uses `PydanticOutputParser` to enforce schema compliance
- If parsing fails, exception is caught and state error is set, triggering replanning

---

## 6. Configuration

### 6.1 Environment Variables (`.env` / `.env.example`)
```bash
# LLM Provider
OPENAI_API_KEY=your_key_here              # Required for OpenAI
# ANTHROPIC_API_KEY=...                   # Future: Anthropic support

# Application Settings
LOG_LEVEL=INFO                            # Logging level
DEBUG=false                               # Debug mode flag
```

### 6.2 MCP Server Configuration (`config/mcp_servers.json`)
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "asterism.servers.filesystem"],
      "transport": "stdio",
      "enabled": true,
      "tools": ["list_files", "read_file", "write_file"]
    },
    "code_parser": {
      "command": "python",
      "args": ["-m", "asterism.servers.code_parser"],
      "transport": "stdio",
      "enabled": true
    }
  }
}
```

**Server Configuration Fields:**
- `command`: Executable to start the server
- `args`: List of arguments (optional, default `[]`)
- `transport`: One of `"stdio"`, `"http_stream"`, `"sse"` (default `"stdio"`)
- `enabled`: Boolean flag to enable/disable server (default `true`)
- `tools`: Optional explicit tool list (if omitted, discovered via `list_tools()`)

### 6.3 Project Configuration (`pyproject.toml`)
- **Dependencies:** Managed by `uv` (fast Python package manager)
- **Ruff Linting:** Line length 120, target Python 3.13, checks E, F, I, UP
- **Pytest:** Test discovery in `tests/` directory, log level INFO

---

## 7. Error Handling

### 7.1 Error Propagation Strategy
- All nodes catch exceptions and set `state["error"]` field
- Errors trigger routing back to `planner_node` for recovery/replanning
- Final state always includes either `final_response` or `error`

### 7.2 Specific Error Scenarios

**Planning Failures:**
- LLM invocation error → state error set → replan
- Structured output parsing failure → caught in planner → state error

**Execution Failures:**
- Missing plan → error "No plan to execute"
- Unsatisfied dependencies → error with list of missing deps
- MCP tool execution failure → `TaskResult.success=False`, error message captured
- LLM-only task failure → exception caught, `TaskResult.success=False`

**MCP Executor Errors:**
- Server not found → `ValueError` with "No metadata found"
- Server disabled → returns `success=False` with "server not enabled"
- Tool not found → returns `success=False` with "tool not found"
- Transport failure → exception caught, returns `success=False`

**Finalizer Errors:**
- LLM response generation failure → fallback message with error details
- Always produces `AgentResponse` (never fails outright)

### 7.3 Checkpoint Storage
- SQLite database at `.checkpoints/agent.db` (configurable via `db_path`)
- Automatic directory creation
- Session cleanup via `clear_session(session_id)` (deletes from `checkpoints` table)

---

## 8. Key Patterns

### 8.1 Dependency Injection via Closures
The `Agent` class uses closure factories to inject dependencies into nodes:
```python
def _make_executor_node(self):
    llm = self.llm
    mcp_executor = self.mcp_executor
    def _executor_node(state: AgentState) -> AgentState:
        return executor_node(llm, mcp_executor, state)
    return _executor_node
```
This avoids global state and keeps nodes pure functions.

### 8.2 Configuration-Driven Execution
MCP tool routing is entirely configuration-based:
- Server metadata loaded from JSON
- Transport instantiated dynamically via `create_transport()`
- Tools discovered at runtime via `list_tools()`
- Enables adding new servers without code changes

### 8.3 State Immutability Pattern
All node functions follow:
```python
def node_fn(..., state: AgentState) -> AgentState:
    new_state = state.copy()
    # modify new_state
    return new_state
```
This ensures LangGraph can properly track state transitions.

### 8.4 Singleton Pattern for Global Access
Both `MCPExecutor` and `MCPConfig` use module-level singletons:
```python
_mcp_executor: MCPExecutor | None = None
def get_mcp_executor() -> MCPExecutor:
    global _mcp_executor
    if _mcp_executor is None:
        _mcp_executor = MCPExecutor()
    return _mcp_executor
```
Provides convenient global access while allowing custom instances.

### 8.5 Abstract Factory for Transports
`transport_executor/__init__.py` likely contains `create_transport(transport_type: str) -> BaseTransport` factory function that instantiates the appropriate transport class based on the `transport` config value.

---

## 9. Testing

### 9.1 Test Architecture
Tests mirror the source structure:
```
tests/
├── unit/                    # Isolated component tests
│   ├── agent/
│   │   ├── test_agent.py       # Agent class methods
│   │   └── test_nodes.py       # Individual node functions
│   ├── mcp/
│   │   ├── test_config.py      # MCPConfig loading/validation
│   │   ├── test_executor.py    # MCPExecutor logic
│   │   └── transport_executor/ # Transport implementations
│   └── llm/                    # LLM provider tests (unit/mocks)
│
└── integration_tests/      # Multi-component tests
    ├── conftest.py
    ├── llm/
    │   └── test_openai_provider.py  # Real API calls (if configured)
    └── transport_executor/
        ├── test_stdio.py
        ├── test_http_stream.py
        └── test_sse.py
```

### 9.2 Testing Standards (from `.kilocode/rules/TESTS.md`)
- **Unit Tests:** Cover both success and failure scenarios, use mocks for external dependencies
- **Integration Tests:** Focus on "happy path" only, run against real environment
- **Style:** Functional Python style (`def test_...():`), avoid classes
- **Execution:** Run targeted tests (`pytest path/to/test.py`) during development

### 9.3 Current Test Coverage
- **Agent:** `test_agent.py` likely tests `invoke()`, `clear_session()`, `close()`
- **Nodes:** `test_nodes.py` tests planner, executor, evaluator, finalizer with mocked dependencies
- **MCP Config:** `test_config.py` tests JSON loading, server lookup, enabled filters
- **MCP Executor:** `test_executor.py` tests tool execution, validation, caching
- **Transports:** Unit tests for each transport type (mocked subprocess/HTTP)
- **LLM:** Tests for OpenAI provider (mocked and integration)

---

## 10. Deployment

### 10.1 Build Process
1. **Dependency Installation:**
   ```bash
   uv sync          # Install from pyproject.toml
   uv lock          # Update lock file (if dependencies changed)
   ```

2. **Code Quality:**
   ```bash
   ./auto_format_ruff.sh   # Auto-format with Ruff
   ```

3. **Testing:**
   ```bash
   pytest tests/unit/agent/              # Run specific tests
   pytest                               # Run all tests
   ```

### 10.2 Runtime Deployment
The agent is a library/framework, not a standalone service. Typical usage:

```python
from asterism.agent.agent import Agent
from asterism.llm.openai_provider import OpenAIProvider
from asterism.mcp.executor import MCPExecutor

llm = OpenAIProvider(model="gpt-4", api_key=...)
mcp_executor = MCPExecutor(config_path="config/mcp_servers.json")
agent = Agent(llm=llm, mcp_executor=mcp_executor)

result = agent.invoke(session_id="session1", user_message="Analyze this codebase")
print(result["message"])
```

### 10.3 Production Considerations
- **Checkpoint Database:** Use persistent storage (not ephemeral filesystem)
- **MCP Servers:** Deploy as separate processes or services; ensure they are running and accessible
- **API Keys:** Secure storage (environment variables, secret manager)
- **Logging:** Configure LOG_LEVEL appropriately; consider structured logging
- **Error Monitoring:** Integrate with error tracking (Sentry, etc.) for production failures
- **Resource Cleanup:** Call `agent.close()` on shutdown to release transport connections

---

## 11. Important Notes

### 11.1 Current State & Assumptions
- **Main Entry Point:** `asterism/main.py` is currently empty; needs implementation for specific use cases
- **Skills System:** The README mentions "skills" (persona switching), but the current codebase does NOT implement this. Skills are documented in `workspace/skills/` as future work.
- **MCP Servers:** The `servers/` directory mentioned in README does not exist in the current codebase. MCP servers are external and configured via JSON.
- **Configuration Path:** `MCPConfig` expects `config/mcp_servers.json` relative to project root. Ensure this file exists at runtime.

### 11.2 Known Limitations
- **Single LLM Provider per Agent:** Agent instantiation requires one LLM provider; no dynamic switching mid-workflow
- **No Parallel Task Execution:** Tasks execute sequentially; dependencies are checked but no parallelization
- **Transport Lifecycle:** Transports are started lazily on first use and never stopped until `shutdown()`. Long-running agents may leak resources if servers are restarted.
- **Checkpoint Size:** SQLite checkpoints can grow indefinitely; implement cleanup strategy for production
- **Error Recovery:** Replanning on error may produce same plan → infinite loop. No backoff or plan variation logic.

### 11.3 Design Decisions
- **TypedDict vs Pydantic for State:** Using `TypedDict` for LangGraph compatibility (required type), while internal models use Pydantic for validation
- **Global Singletons:** Convenience trade-off vs testability; tests can still inject custom instances
- **Configuration over Convention:** MCP servers fully configurable via JSON; no hardcoded servers
- **Transport Abstraction:** Clean separation allows adding new protocols without modifying executor

### 11.4 Future Roadmap (from README)
- Implement skill-based routing (persona switching)
- Add more LLM providers (Anthropic, local models)
- Create built-in MCP servers (filesystem, code_parser)
- Develop demo applications (code analysis, report generation)

### 11.5 Contributing Guidelines
- Follow existing patterns: pure node functions, dependency injection, error handling
- Write unit tests for new functionality, integration tests for multi-component features
- Use `uv` for dependency management, `ruff` for linting/formatting
- Update this BLUEPRINT.md when architecture changes

---

*Generated by AI based on codebase analysis on 2025-02-05. Verify against actual implementation.*