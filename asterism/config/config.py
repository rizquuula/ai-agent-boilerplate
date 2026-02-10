"""Configuration loader for Asterism.

Loads configuration from workspace/config.yaml with support for
environment variable resolution (values prefixed with 'env.').
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent metadata configuration."""

    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    description: str = Field(..., description="Agent description")


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(..., description="API host address")
    port: int = Field(..., description="API port number")
    debug: bool = Field(..., description="Debug mode flag")


class ModelProvider(BaseModel):
    """LLM provider configuration."""

    type: str = Field(..., description="Provider type (e.g., openai-compatible)")
    name: str = Field(..., description="Provider name")
    base_url: str | None = Field(default=None, description="Base URL for API")
    api_key: str | None = Field(default=None, description="API key (supports env. prefix)")


class ModelsConfig(BaseModel):
    """Models configuration section."""

    provider: list[ModelProvider] = Field(..., description="List of model providers")
    default: str = Field(..., description="Default model to use")
    fallback: list[str] = Field(default_factory=list, description="Fallback models")


class ConfigData(BaseModel):
    """Complete configuration data structure."""

    agent: AgentConfig = Field(..., description="Agent metadata")
    api: APIConfig = Field(..., description="API configuration")
    models: ModelsConfig = Field(..., description="Models configuration")


class Config:
    """Global configuration singleton for Asterism.

    Loads configuration from workspace/config.yaml with environment variable
    resolution. Values prefixed with 'env.' are automatically resolved from
    environment variables.

    Example:
        api_key: env.OPENROUTER_API_KEY  # Loads from os.getenv("OPENROUTER_API_KEY")

    Usage:
        from asterism.config import Config

        config = Config()  # Uses default workspace path
        print(config.data.agent.name)
        print(config.get_model_provider("openrouter"))
    """

    _instance: "Config | None" = None

    def __new__(cls, workspace_path: str | None = None) -> "Config":
        """Create or return the singleton instance.

        Args:
            workspace_path: Path to workspace directory. If None, uses WORKSPACE_DIR
                         environment variable or defaults to './workspace'.
                         Only used on first initialization.

        Returns:
            Config: The singleton configuration instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, workspace_path: str | None = None) -> None:
        """Initialize the configuration (only runs once).

        Args:
            workspace_path: Path to workspace directory. Ignored after first init.
        """
        if self._initialized:
            return

        self._workspace_path = self._resolve_workspace_path(workspace_path)
        self._config_file = Path(self._workspace_path) / "config.yaml"
        self._raw_config: dict[str, Any] = {}
        self._data: ConfigData | None = None

        self._load()
        self._initialized = True

    def _resolve_workspace_path(self, workspace_path: str | None) -> str:
        """Resolve the workspace path from parameter or environment.

        Args:
            workspace_path: Explicit path provided, or None to auto-resolve.

        Returns:
            str: Resolved workspace path.
        """
        if workspace_path is not None:
            return workspace_path

        env_path = os.getenv("WORKSPACE_DIR")
        if env_path is not None:
            return env_path

        return "./workspace"

    def _load(self) -> None:
        """Load and parse the configuration file."""
        if not self._config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._config_file}")

        with open(self._config_file, encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f) or {}

        # Resolve environment variables in the raw config
        resolved_config = self._resolve_env_values(self._raw_config)

        # Parse into Pydantic models
        self._data = ConfigData.model_validate(resolved_config)

    def _resolve_env_values(self, data: Any) -> Any:
        """Recursively resolve environment variable references.

        Any string value starting with 'env.' is resolved from the
        corresponding environment variable. If not found, defaults to None.

        Args:
            data: The data structure to process (dict, list, or scalar).

        Returns:
            The data structure with env. values resolved.
        """
        if isinstance(data, dict):
            return {key: self._resolve_env_values(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._resolve_env_values(item) for item in data]

        if isinstance(data, str) and data.startswith("env."):
            env_var = data[4:]  # Remove 'env.' prefix
            return os.getenv(env_var)

        return data

    @property
    def data(self) -> ConfigData:
        """Get the parsed configuration data.

        Returns:
            ConfigData: The configuration data model.

        Raises:
            RuntimeError: If configuration hasn't been loaded.
        """
        if self._data is None:
            raise RuntimeError("Configuration not loaded")
        return self._data

    @property
    def workspace_path(self) -> str:
        """Get the workspace path.

        Returns:
            str: Path to the workspace directory.
        """
        return self._workspace_path

    def get_model_provider(self, name: str) -> ModelProvider | None:
        """Get a model provider by name.

        Args:
            name: The provider name to look up.

        Returns:
            ModelProvider | None: The provider if found, None otherwise.
        """
        for provider in self.data.models.provider:
            if provider.name == name:
                return provider
        return None

    def get_default_model_provider(self) -> ModelProvider | None:
        """Get the default model provider.

        The default model format is 'provider_name/model_path'. This method
        extracts the provider name from the default model string.

        Returns:
            ModelProvider | None: The default provider if found.
        """
        default_model = self.data.models.default
        if "/" not in default_model:
            return None

        provider_name = default_model.split("/")[0]
        return self.get_model_provider(provider_name)

    def reload(self) -> None:
        """Reload the configuration from disk.

        Useful for picking up changes without restarting the application.
        """
        self._load()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing).

        After calling this, the next Config() call will create a new instance.
        """
        cls._instance = None
