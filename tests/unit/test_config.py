"""Tests for configuration module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    ModelParams,
    PathsSettings,
    TrainingSettings,
    get_default_config,
    load_config,
)


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_config(self) -> None:
        """Test creating config with default values."""
        config = get_default_config()

        assert config.project.name == "ArchRef ML"
        assert config.project.version == "0.1.0"
        assert config.task.type == "classification"
        assert config.model.name == "RandomForest"

    def test_load_config_from_file(self, config_file: Path) -> None:
        """Test loading config from YAML file."""
        config = load_config(config_file)

        assert config.project.name == "Test Project"
        assert config.task.type == "classification"
        assert config.model.params.n_estimators == 10

    def test_load_config_file_not_found(self) -> None:
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestModelParams:
    """Tests for ModelParams validation."""

    def test_valid_params(self) -> None:
        """Test valid model parameters."""
        params = ModelParams(n_estimators=100, max_depth=10)
        assert params.n_estimators == 100
        assert params.max_depth == 10

    def test_n_estimators_minimum(self) -> None:
        """Test n_estimators must be at least 1."""
        with pytest.raises(ValidationError):
            ModelParams(n_estimators=0)

    def test_n_estimators_maximum(self) -> None:
        """Test n_estimators must be at most 10000."""
        with pytest.raises(ValidationError):
            ModelParams(n_estimators=20000)

    def test_extra_params_allowed(self) -> None:
        """Test that extra parameters are allowed."""
        params = ModelParams(n_estimators=100, custom_param="value")
        assert params.model_dump()["custom_param"] == "value"


class TestTrainingSettings:
    """Tests for TrainingSettings validation."""

    def test_valid_settings(self) -> None:
        """Test valid training settings."""
        settings = TrainingSettings(test_size=0.3, cv_folds=10)
        assert settings.test_size == 0.3
        assert settings.cv_folds == 10

    def test_test_size_bounds(self) -> None:
        """Test test_size must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TrainingSettings(test_size=0)

        with pytest.raises(ValidationError):
            TrainingSettings(test_size=1)

    def test_cv_folds_minimum(self) -> None:
        """Test cv_folds must be at least 2."""
        with pytest.raises(ValidationError):
            TrainingSettings(cv_folds=1)


class TestPathsSettings:
    """Tests for PathsSettings."""

    def test_string_to_path_conversion(self) -> None:
        """Test that string paths are converted to Path objects."""
        settings = PathsSettings(data_raw="custom/path")
        assert isinstance(settings.data_raw, Path)
        assert settings.data_raw == Path("custom/path")
