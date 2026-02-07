# Asterism Agent - Project Blueprint

## 1. Overview

High-performance AI agent framework built with **LangGraph** for workflow orchestration and **MCP** for tool execution.

**Core Features:**
- Plan-Execute-Evaluate cycle with automatic retry logic
- Dynamic LLM provider abstraction (OpenAI)
- Multi-transport MCP tool execution (stdio, http_stream, SSE)
- State persistence via SQLite checkpointing
- Runtime system prompt loading (SOUL.md + AGENT.md)
- **Robust JSON parsing with retry logic and markdown extraction**
- **Comprehensive logging across all nodes for debugging**

## 2. Directory Structure

```
asterism/
├── agent/           # Core agent implementation
│   ├── agent.py    # LangGraph workflow orchestration
│   ├── nodes/      # Planner, Executor, Evaluator, Finalizer
│   │   ├── planner/
│   │   │   ├── node.py      # Plan creation with validation
│   │   │   ├── prompts.py   # Enhanced prompts with JSON examples
│   │   │   └── utils.py     # Tool context formatting
│   │   ├── executor/
│   │   │   └── node.py      # Task execution with logging
│   │   ├── evaluator/
│   │   │   ├── node.py      # Decision making with detailed context
│   │   │   └── utils.py     # Task input resolution
│   │   └── finalizer/
│   │       └── node.py      # Response generation with logging
│   ├── state/      # AgentState TypedDict
│   └── models/     # Pydantic schemas (Task, Plan, TaskResult)
├── core/           # Core utilities
│   └── prompt_loader.py  # SOUL.md/AGENT.md loader
├── llm/            # LLM providers
│   ├── base.py     # BaseLLMProvider abstract interface
│   └── openai_provider.py  # OpenAI implementation with JSON mode & retry
└── mcp/            # MCP integration
    ├── config.py    # MCP server configuration
    ├── executor.py  # Dynamic tool executor
    └── transport_executor/  # stdio, http_stream, sse

tests/
├── unit/           # Unit tests (93 tests, ~100% coverage)
└── integration_tests/  # Integration tests

workspace/
├── AGENT.md        # Agent identity & capabilities (mandatory)
├── SOUL.md         # Core values & philosophy (mandatory)
└── PERSONALITY.md  # Agent personality configuration
```

## 3. Architecture

### Agent Workflow (LangGraph State Machine)

```
START → Planner → Executor → Evaluator → (loop or) Finalizer → END
              ↑              ↓              ↓
              └──────────────┴──────────────┘
```

### System Prompt Loading

Every LLM call receives:
```
[SystemMessage(content=SOUL.md + AGENT.md)]
[SystemMessage(content=node-specific instructions)]
[HumanMessage(content=user request)]
```

**Flow:**
1. `SystemPromptLoader.load()` reads `workspace/SOUL.md` and `workspace/AGENT.md` fresh
2. Content is combined and passed to LLM provider
3. `FileNotFoundError` raised if either file is missing (mandatory)

### LLM Provider Interface

```python
class BaseLLMProvider(ABC):
    def __init__(self, prompt_loader: SystemPromptLoader | None = None)
    def invoke(prompt: str | list[BaseMessage], **kwargs) -> str
    def invoke_structured(prompt: str | list[BaseMessage], schema: type, max_retries: int = 3, **kwargs) -> Any
```

**Key Features:**
- **JSON Mode**: OpenAI provider uses `response_format={"type": "json_object"}` for structured output
- **Retry Logic**: Up to 3 retries with exponential backoff for parsing failures
- **Markdown Extraction**: Automatically extracts JSON from markdown code blocks if parser fails
- **Detailed Error Messages**: Includes raw LLM output in error messages for debugging

**Message Types:** Uses LangChain's `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`

## 4. Data Models

| Model | Purpose |
|-------|---------|
| `Task` | Atomic action with tool_call, tool_input, depends_on |
| `Plan` | Ordered list of Tasks with reasoning |
| `TaskResult` | Execution result (success/error/result) |
| `AgentResponse` | Final user-facing response with execution trace |
| `EvaluationResult` | Decision (continue/replan/finalize) with reasoning |
| `LLMUsage` | Token usage tracking per node |

## 5. Key Patterns

1. **Dependency Injection** - Agent injects LLM/MCP into nodes via closures
2. **State Immutability** - Nodes return new state: `state.copy()`
3. **Configuration-Driven** - MCP servers defined in JSON, no code changes needed
4. **Singleton Access** - Global `get_mcp_executor()` for convenience
5. **Comprehensive Logging** - All nodes use Python logging for debug/info/warning/error
6. **Error Context Propagation** - Errors include detailed context for replanning

## 6. Configuration

**Environment:** `OPENAI_API_KEY` (required), `LOG_LEVEL`, `DEBUG`

**MCP Servers:** `workspace/mcp_servers/mcp_servers.json`
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "transport": "stdio"
    },
    "localtime_mcp": {
      "command": "uvx",
      "args": ["mcp_servers/localtime_mcp"],
      "transport": "stdio"
    }
  }
}
```

## 7. Error Handling

- All nodes catch exceptions → set `state["error"]`
- Errors trigger replanning (route to Planner)
- Final state always has `final_response` or `error`
- Missing SOUL.md/AGENT.md → `FileNotFoundError` (agent cannot run)
- **JSON Parsing Errors**: Retry with exponential backoff, extract from markdown if needed
- **Planning Failures**: Error context added to messages for replanning
- **Execution Failures**: Detailed error info passed to evaluator for decision making

## 8. Planner Node Enhancements

### JSON Output Instructions
The planner now includes:
- **Explicit JSON format requirements** in system prompt
- **Example valid JSON output** showing complete plan structure
- **Workspace context** explaining directory structure
- **File discovery guidance** using `filesystem:search_files` before editing

### File Editing Workflow
For file operations, the planner now creates tasks in this order:
1. **Search**: Use `filesystem:search_files` to locate target files
2. **Read**: Use `filesystem:read_file` to get current content
3. **Edit**: Use `filesystem:write_file` to save changes
4. **Verify**: Optional verification task

## 9. Testing

- **Unit Tests:** 93 tests covering all components
- **Style:** Functional Python (`def test_...()`), no classes
- **Coverage:** ~100% for core modules
- **Execution:** `uv run pytest tests/unit/`

## 10. Commands

```bash
uv sync          # Install dependencies
uv run pytest    # Run tests
./auto_format_ruff.sh  # Format code
```

## 11. Recent Improvements

### Issue 1: Plan Parsing Errors (FIXED)
- Added JSON mode enforcement in OpenAI provider
- Implemented retry logic with exponential backoff (3 attempts)
- Added markdown code block extraction as fallback
- Enhanced error messages with raw LLM output
- Added plan validation (must have tasks)

### Issue 2: File Editing Failures (FIXED)
- Enhanced planner prompts with workspace context
- Added file discovery step using `filesystem:search_files`
- Included example JSON showing file editing workflow
- Added guidance on directory structure and paths

### Additional Improvements
- Added comprehensive logging to all nodes (planner, executor, evaluator, finalizer)
- Enhanced evaluator replanning context with detailed error information
- Improved error propagation through state messages
- Better task input resolution with logging
