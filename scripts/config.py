"""Configuration management for the application."""

import tomllib
import os
from pathlib import Path


def load_config():
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_roboflow_api_key(config: dict) -> str:
    """Get Roboflow API key from config or environment variable.
    
    Checks in order:
    1. config.toml (local development)
    2. ROBOFLOW_API_KEY environment variable (production/Fly.io)
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        API key string or None if not configured
    """
    # First try config.toml
    api_key = config.get("api", {}).get("roboflow_api_key")
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("Warning: roboflow_api_key not configured. Fishing model will not be available.")
        print("  - Local: Set in config.toml [api] section")
        print("  - Production: Set via 'fly secrets set ROBOFLOW_API_KEY=...'")
    
    return api_key
