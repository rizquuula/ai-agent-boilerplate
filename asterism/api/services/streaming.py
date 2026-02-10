"""SSE streaming implementation for chat completions."""

import json
import time
from collections.abc import AsyncGenerator

from ..models import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
)


async def stream_chat_completion(
    request_id: str,
    model: str,
    agent_service,
    request,
) -> AsyncGenerator[str]:
    """Stream chat completion as SSE events.

    Args:
        request_id: Unique request identifier
        model: Model identifier
        agent_service: Agent service instance
        request: Chat completion request

    Yields:
        SSE-formatted event strings
    """
    # Send start event with role
    start_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None,
            )
        ],
    )
    yield f"data: {json.dumps(start_chunk.model_dump())}\n\n"

    # Stream content tokens
    full_content = ""
    async for token, metadata in agent_service.run_streaming(request, request_id):
        if metadata is not None:
            # Final metadata received - send finish event
            finish_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {json.dumps(finish_chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
            break
        else:
            # Regular token
            full_content += token
            content_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"content": token},
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {json.dumps(content_chunk.model_dump())}\n\n"
