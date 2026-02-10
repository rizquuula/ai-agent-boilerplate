"""Pydantic models for OpenAI-compatible API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # For tool messages


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions."""

    model: str = Field(..., description="Model ID (provider/model format)")
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None

    # Asterism-specific extensions
    session_id: str | None = Field(default=None, description="Optional session ID for tracing")


class ChatCompletionStreamChoice(BaseModel):
    """Single choice in streaming response."""

    index: int = 0
    delta: dict  # {role?: string, content?: string}
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """SSE streaming response chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int  # Unix timestamp
    model: str
    choices: list[ChatCompletionStreamChoice]


class ChatCompletionMessage(BaseModel):
    """Message in non-streaming response."""

    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    """Single choice in non-streaming response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Non-streaming response body."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    """Model information for /v1/models."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsListResponse(BaseModel):
    """Response for /v1/models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthStatus(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    version: str
    providers: dict[str, str]


class ErrorDetail(BaseModel):
    """Error detail for API errors."""

    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    """Error response body."""

    error: ErrorDetail
