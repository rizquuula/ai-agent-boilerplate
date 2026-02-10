"""FastAPI dependency injection."""

from fastapi import Depends, Header

from asterism.config import Config
from asterism.llm import LLMProviderRouter
from asterism.mcp.config import MCPConfigLoader
from asterism.mcp.executor import MCPExecutor

from .exceptions import AuthenticationError


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()


def get_api_key(
    authorization: str | None = Header(None),
    config: Config = Depends(get_config),
) -> str:
    """Validate and return API key from Authorization header.

    Args:
        authorization: The Authorization header value
        config: Configuration instance

    Returns:
        The validated API key

    Raises:
        AuthenticationError: If the API key is invalid or missing
    """
    valid_keys = config.get_api_keys()

    # If no API keys are configured, allow all requests (development mode)
    if not valid_keys:
        return "dev-key"

    if not authorization:
        raise AuthenticationError("Missing Authorization header")

    # Parse "Bearer <token>" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError("Invalid Authorization header format. Use 'Bearer <api_key>'")

    api_key = parts[1]

    if api_key not in valid_keys:
        raise AuthenticationError("Invalid API key")

    return api_key


def get_llm_router(config: Config = Depends(get_config)) -> LLMProviderRouter:
    """Get the LLM provider router instance.

    Args:
        config: Configuration instance

    Returns:
        Configured LLMProviderRouter
    """
    return LLMProviderRouter(config)


def get_mcp_executor(config: Config = Depends(get_config)) -> MCPExecutor:
    """Get the MCP executor instance.

    Args:
        config: Configuration instance

    Returns:
        Configured MCPExecutor
    """
    servers_file = config.get_mcp_servers_file()
    mcp_config = MCPConfigLoader.load(servers_file)
    return MCPExecutor(mcp_config)
