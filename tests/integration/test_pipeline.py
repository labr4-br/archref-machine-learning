"""Integration tests for training pipeline."""

from pathlib import Path

import pytest

from src.pipeline import run_pipeline


class TestPipeline:
    """Integration tests for the training pipeline."""

    @pytest.fixture
    def pipeline_config(self, temp_dir: Path, config_yaml_content: str) -> Path:
        """Create config file with temp paths."""
        config_content = f"""
project:
  name: "Test Pipeline"
  version: "0.1.0"

task:
  type: "classification"

paths:
  data_raw: "{temp_dir}/data/raw"
  data_processed: "{temp_dir}/data/processed"
  models: "{temp_dir}/models"
  logs: "{temp_dir}/logs/app.log"

model:
  name: "RandomForest"
  params:
    n_estimators: 10
    random_state: 42
    max_depth: 5

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 3

integrations:
  mlflow:
    enabled: false
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)
        return config_path

    def test_pipeline_runs_successfully(self, pipeline_config: Path) -> None:
        """Test that pipeline runs without errors."""
        result = run_pipeline(pipeline_config)

        assert result is not None
        assert "model_path" in result
        assert "metrics" in result
        assert "config" in result

    def test_pipeline_creates_model_file(
        self,
        pipeline_config: Path,
        temp_dir: Path,
    ) -> None:
        """Test that pipeline creates model file."""
        run_pipeline(pipeline_config)

        model_path = temp_dir / "models" / "RandomForest.pkl"
        assert model_path.exists()

    def test_pipeline_returns_metrics(self, pipeline_config: Path) -> None:
        """Test that pipeline returns valid metrics."""
        result = run_pipeline(pipeline_config)

        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # Metrics should be in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_pipeline_creates_log_file(
        self,
        pipeline_config: Path,
        temp_dir: Path,
    ) -> None:
        """Test that pipeline creates log file."""
        run_pipeline(pipeline_config)

        log_path = temp_dir / "logs" / "app.log"
        assert log_path.exists()

        # Log should contain pipeline messages
        log_content = log_path.read_text()
        assert "Training Pipeline" in log_content

    def test_pipeline_with_invalid_config(self, temp_dir: Path) -> None:
        """Test pipeline fails with invalid config."""
        with pytest.raises(FileNotFoundError):
            run_pipeline(temp_dir / "nonexistent.yaml")
