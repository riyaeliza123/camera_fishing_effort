"""Configuration management for the application."""

import tomllib
from pathlib import Path


def load_config():
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_roboflow_api_key(config: dict) -> str:
    """Get Roboflow API key from config.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        API key string or None if not configured
    """
    api_key = config.get("api", {}).get("roboflow_api_key")
    if not api_key:
        print("Warning: roboflow_api_key not found in config.toml. Fishing model will not be available.")
    return api_key
