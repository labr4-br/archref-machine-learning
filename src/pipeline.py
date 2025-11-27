"""Training pipeline for ML ArchRef."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sklearn.model_selection import train_test_split

from src.config import AppConfig, load_config
from src.data.mock_loader import MockDataLoader
from src.exceptions import PipelineStepError
from src.integrations.mlflow_tracker import MLflowTracker
from src.tasks.classification.metrics import ClassificationMetrics
from src.tasks.classification.models import RandomForestModel
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.core.base_loader import BaseDataLoader
    from src.core.base_metrics import BaseMetrics
    from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


def get_model(config: AppConfig) -> BaseModel:
    """Get the appropriate model based on config.

    Args:
        config: Application configuration.

    Returns:
        Initialized model instance.

    Raises:
        PipelineStepError: If model name is not supported.
    """
    model_name = config.model.name
    params = config.model.params.model_dump()

    if model_name == "RandomForest":
        return RandomForestModel(params)
    else:
        raise PipelineStepError(
            "model_initialization",
            f"Unknown model: {model_name}. Supported: RandomForest",
        )


def get_metrics(config: AppConfig) -> BaseMetrics:
    """Get the appropriate metrics calculator based on task type.

    Args:
        config: Application configuration.

    Returns:
        Metrics calculator instance.

    Raises:
        PipelineStepError: If task type is not supported.
    """
    task_type = config.task.type

    if task_type == "classification":
        return ClassificationMetrics()
    else:
        raise PipelineStepError(
            "metrics_initialization",
            f"Unknown task type: {task_type}. Supported: classification",
        )


def get_data_loader(config: AppConfig) -> BaseDataLoader:
    """Get the appropriate data loader.

    Args:
        config: Application configuration.

    Returns:
        Data loader instance.
    """
    return MockDataLoader(config.model_dump())


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Run the complete training pipeline.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary with training results including metrics.

    Raises:
        PipelineStepError: If any pipeline step fails.
    """
    # 1. Load Configuration
    config = load_config(config_path)
    pipeline_logger = setup_logger("training_pipeline", str(config.paths.logs))

    pipeline_logger.info("=" * 60)
    pipeline_logger.info(f"Starting Training Pipeline: {config.project.name}")
    pipeline_logger.info(f"Task Type: {config.task.type}")
    pipeline_logger.info(f"Model: {config.model.name}")
    pipeline_logger.info("=" * 60)

    # 2. Initialize MLflow Tracker
    tracker = MLflowTracker(config.integrations.mlflow)

    # 3. Load and Process Data
    pipeline_logger.info("Loading data...")
    try:
        loader = get_data_loader(config)
        df = loader.load_and_process()
        pipeline_logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        raise PipelineStepError("data_loading", str(e)) from e

    # 4. Prepare Features and Target
    X = df.drop("target", axis=1)
    y = df["target"]

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
    )
    pipeline_logger.info(
        f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}"
    )

    # 6. Initialize Model and Metrics
    model = get_model(config)
    metrics_calculator = get_metrics(config)

    # 7. Train and Evaluate with MLflow Tracking
    with tracker.start_run(run_name=f"{config.model.name}-training"):
        # Log configuration
        tracker.log_params(
            {
                "model_name": config.model.name,
                "task_type": config.task.type,
                **config.model.params.model_dump(),
                "test_size": config.training.test_size,
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1],
            }
        )

        # Train
        pipeline_logger.info("Training model...")
        try:
            model.train(X_train, y_train)
        except Exception as e:
            raise PipelineStepError("training", str(e)) from e

        # Predict
        pipeline_logger.info("Generating predictions...")
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        # Calculate Metrics
        pipeline_logger.info("Evaluating model...")
        metrics = metrics_calculator.calculate(y_test, y_pred, y_proba=y_proba)
        flat_metrics = metrics_calculator.get_flat_metrics(y_test, y_pred, y_proba)

        # Log metrics
        tracker.log_metrics(flat_metrics)

        pipeline_logger.info("-" * 40)
        pipeline_logger.info("Evaluation Results:")
        for name, value in flat_metrics.items():
            pipeline_logger.info(f"  {name}: {value:.4f}")
        pipeline_logger.info("-" * 40)

        # Save Model
        model_dir = config.paths.models
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{config.model.name}.pkl"

        model.save(model_path)
        pipeline_logger.info(f"Model saved to: {model_path}")

        # Log model to MLflow
        tracker.log_model(model, "model")

    pipeline_logger.info("=" * 60)
    pipeline_logger.info("Pipeline completed successfully!")
    pipeline_logger.info("=" * 60)

    return {
        "model_path": str(model_path),
        "metrics": flat_metrics,
        "config": config.model_dump(),
    }


def main() -> None:
    """CLI entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="ML ArchRef Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline
  python -m src.pipeline --config custom_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
