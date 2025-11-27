"""Pydantic configuration models with validation."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseModel):
    """Project metadata settings."""

    name: str = "ArchRef ML"
    version: str = "0.1.0"


class TaskSettings(BaseModel):
    """Task type configuration."""

    type: Literal["classification", "regression", "clustering", "anomaly", "timeseries"] = (
        "classification"
    )


class PathsSettings(BaseModel):
    """File paths configuration."""

    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    models: Path = Path("models")
    logs: Path = Path("reports/logs/app.log")

    @field_validator("data_raw", "data_processed", "models", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)


class ModelParams(BaseModel):
    """Model hyperparameters."""

    n_estimators: int = Field(default=100, ge=1, le=10000)
    random_state: int = Field(default=42)
    max_depth: int | None = Field(default=10, ge=1)

    model_config = {"extra": "allow"}  # Allow additional params


class ModelSettings(BaseModel):
    """Model configuration."""

    name: str = "RandomForest"
    params: ModelParams = Field(default_factory=ModelParams)


class TrainingSettings(BaseModel):
    """Training configuration."""

    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42)
    cv_folds: int = Field(default=5, ge=2, le=20)


class MLflowSettings(BaseModel):
    """MLflow integration settings."""

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "default"
    run_name: str | None = None


class IntegrationsSettings(BaseModel):
    """External integrations configuration."""

    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)


class AppConfig(BaseSettings):
    """Main application configuration.

    This class loads configuration from:
    1. Default values
    2. YAML config file (if provided)
    3. Environment variables (with ML_ prefix)

    Environment variables override YAML values.
    Example: ML_PROJECT__NAME="My Project"
    """

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    task: TaskSettings = Field(default_factory=TaskSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    integrations: IntegrationsSettings = Field(default_factory=IntegrationsSettings)

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If configuration is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        yaml_config: dict[str, Any] = yaml.safe_load(f) or {}

    # Create config with YAML values, env vars will override automatically
    return AppConfig(**yaml_config)


def get_default_config() -> AppConfig:
    """Get configuration with default values only.

    Useful for testing or when no config file is available.

    Returns:
        AppConfig instance with defaults.
    """
    return AppConfig()
