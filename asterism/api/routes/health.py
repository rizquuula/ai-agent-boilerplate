"""Health check endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from asterism.config import Config
from asterism.llm import LLMProviderRouter

from ..dependencies import get_config, get_llm_router
from ..models import HealthStatus

router = APIRouter()


@router.get("/health")
async def health_check(
    config: Annotated[Config, Depends(get_config)],
    llm_router: Annotated[LLMProviderRouter, Depends(get_llm_router)],
) -> HealthStatus:
    """Health check endpoint.

    Returns the health status of the API and all configured providers.

    Args:
        config: Configuration instance
        llm_router: LLM provider router

    Returns:
        Health status information
    """
    # Check provider availability
    providers: dict[str, str] = {}
    all_healthy = True

    for provider_config in config.data.models.provider:
        provider = llm_router.providers.get(provider_config.name)
        if provider:
            providers[provider_config.name] = "available"
        else:
            providers[provider_config.name] = "unavailable"
            all_healthy = False

    return HealthStatus(
        status="healthy" if all_healthy else "unhealthy",
        version=config.data.agent.version,
        providers=providers,
    )
