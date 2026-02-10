"""FastAPI application factory."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asterism.config import Config

from .exceptions import (
    AllProvidersFailedError,
    APIError,
    all_providers_failed_handler,
    api_error_handler,
    generic_exception_handler,
)
from .routes import chat_router, health_router, models_router

logger = logging.getLogger(__name__)


def create_api_app(config: Config | None = None) -> FastAPI:
    """Factory function to create the FastAPI application.

    Args:
        config: Configuration instance. If None, creates a new Config.

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = Config()

    app = FastAPI(
        title="Asterism API",
        description="OpenAI-compatible API for Asterism Agent",
        version=config.data.agent.version,
        docs_url="/docs" if config.data.api.debug else None,
        redoc_url="/redoc" if config.data.api.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.data.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(AllProvidersFailedError, all_providers_failed_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include routers
    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(health_router, prefix="/v1")

    @app.on_event("startup")
    async def startup_event():
        """Handle application startup."""
        logger.info(f"Asterism API v{config.data.agent.version} starting...")
        logger.info(f"Workspace: {config.workspace_path}")
        logger.info(f"Default model: {config.data.models.default}")
        logger.info(f"Configured providers: {[p.name for p in config.data.models.provider]}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Handle application shutdown."""
        logger.info("Asterism API shutting down...")

    return app
