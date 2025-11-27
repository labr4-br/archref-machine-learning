"""Configuration loader utilities.

This module provides backward compatibility with the legacy config loading.
For new code, use src.config.load_config directly.
"""

from pathlib import Path
from typing import Any

import yaml

# Re-export the new config loader for convenience
from src.config import AppConfig, load_config as load_app_config


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    This function is kept for backward compatibility.
    For new code, use src.config.load_config instead.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration as a dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    return config


__all__ = ["load_config", "load_app_config", "AppConfig"]
