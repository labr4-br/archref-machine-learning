"""Configuration loader utilities.

This module provides backward compatibility with the legacy config loading.
For new code, use src.config.load_config directly.
"""

from pathlib import Path
from typing import Any

import yaml

from src.config import AppConfig
from src.config import load_config as load_app_config


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

    with path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    return config


__all__ = ["AppConfig", "load_app_config", "load_config"]
