"""Chat completions endpoint."""

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from asterism.config import Config
from asterism.llm import LLMProviderRouter
from asterism.mcp.executor import MCPExecutor

from ..dependencies import get_config, get_llm_router, get_mcp_executor
from ..models import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)
from ..services.agent_service import AgentService
from ..services.streaming import stream_chat_completion

router = APIRouter()


def get_agent_service(
    llm_router: Annotated[LLMProviderRouter, Depends(get_llm_router)],
    mcp_executor: Annotated[MCPExecutor, Depends(get_mcp_executor)],
    config: Annotated[Config, Depends(get_config)],
) -> AgentService:
    """Create agent service with dependencies."""
    return AgentService(llm_router, mcp_executor, config)


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    Stateless - each request creates a fresh agent instance.

    Args:
        request: The chat completion request
        agent_service: The agent service instance

    Returns:
        Either a ChatCompletionResponse (non-streaming) or StreamingResponse (streaming)
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        # SSE streaming response
        return StreamingResponse(
            stream_chat_completion(
                request_id=request_id,
                model=request.model,
                agent_service=agent_service,
                request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming response
    result = await agent_service.run_completion(
        request=request,
        request_id=request_id,
    )

    # Extract usage from result
    total_usage = result.get("total_usage", {})
    usage = UsageInfo(
        prompt_tokens=total_usage.get("total_prompt_tokens", 0),
        completion_tokens=total_usage.get("total_completion_tokens", 0),
        total_tokens=total_usage.get("total_tokens", 0),
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=result.get("message", ""),
                ),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
