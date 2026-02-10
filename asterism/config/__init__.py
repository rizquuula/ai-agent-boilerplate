"""Configuration module for Asterism.

This module provides global access to the application configuration
loaded from workspace/config.yaml with environment variable resolution.
"""

from .config import Config, ConfigData

__all__ = ["Config", "ConfigData"]
