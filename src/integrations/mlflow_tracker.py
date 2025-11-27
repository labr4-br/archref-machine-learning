"""MLflow integration for experiment tracking."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import MLflowSettings
    from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper for MLflow experiment tracking.

    Provides a simplified interface for logging experiments, metrics,
    parameters, and models to MLflow. Handles the case when MLflow
    is not installed or disabled gracefully.

    Attributes:
        settings: MLflow configuration settings.
        enabled: Whether MLflow tracking is enabled.
    """

    def __init__(self, settings: MLflowSettings | dict[str, Any] | None = None) -> None:
        """Initialize the MLflow tracker.

        Args:
            settings: MLflow configuration. Can be MLflowSettings instance
                or a dictionary with keys: enabled, tracking_uri, experiment_name.
        """
        self._mlflow = None
        self._active_run = None

        # Handle different input types
        if settings is None:
            self.enabled = False
            self.tracking_uri = "http://localhost:5000"
            self.experiment_name = "default"
            self.run_name = None
        elif isinstance(settings, dict):
            self.enabled = settings.get("enabled", False)
            self.tracking_uri = settings.get("tracking_uri", "http://localhost:5000")
            self.experiment_name = settings.get("experiment_name", "default")
            self.run_name = settings.get("run_name")
        else:
            self.enabled = settings.enabled
            self.tracking_uri = settings.tracking_uri
            self.experiment_name = settings.experiment_name
            self.run_name = settings.run_name

        if self.enabled:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize MLflow client."""
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow initialized: uri={self.tracking_uri}, "
                f"experiment={self.experiment_name}"
            )
        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install archref-ml[mlflow]"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.enabled = False

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[MLflowTracker, None, None]:
        """Context manager for MLflow run.

        Args:
            run_name: Name for the run. Uses settings.run_name if not provided.
            tags: Additional tags for the run.

        Yields:
            Self for method chaining.

        Example:
            with tracker.start_run(run_name="experiment-1"):
                tracker.log_params(params)
                tracker.log_metrics(metrics)
        """
        if not self.enabled or self._mlflow is None:
            logger.debug("MLflow disabled, skipping run context")
            yield self
            return

        run_name = run_name or self.run_name

        try:
            self._active_run = self._mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
            yield self
        finally:
            if self._active_run:
                self._mlflow.end_run()
                logger.info("Ended MLflow run")
                self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values.
        """
        if not self.enabled or self._mlflow is None:
            return

        # Flatten nested dicts and convert values to strings
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            try:
                self._mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values.
            step: Step number for the metrics.
        """
        if not self.enabled or self._mlflow is None:
            return

        for key, value in metrics.items():
            # Only log numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    self._mlflow.log_metric(key, value, step=step)
                except Exception as e:
                    logger.warning(f"Failed to log metric {key}: {e}")

    def log_model(
        self,
        model: BaseModel,
        artifact_path: str = "model",
    ) -> None:
        """Log a model to MLflow.

        Args:
            model: The trained model to log.
            artifact_path: Path within the artifact store.
        """
        if not self.enabled or self._mlflow is None:
            return

        if model.model is None:
            logger.warning("Cannot log untrained model")
            return

        try:
            self._mlflow.sklearn.log_model(model.model, artifact_path)
            logger.info(f"Model logged to MLflow: {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a local file as an artifact.

        Args:
            local_path: Path to the file to log.
        """
        if not self.enabled or self._mlflow is None:
            return

        try:
            self._mlflow.log_artifact(str(local_path))
            logger.info(f"Artifact logged: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        if not self.enabled or self._mlflow is None:
            return

        try:
            self._mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set tag {key}: {e}")

    @staticmethod
    def _flatten_dict(
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten.
            parent_key: Prefix for keys.
            sep: Separator between nested keys.

        Returns:
            Flattened dictionary.
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MLflowTracker(enabled={self.enabled}, "
            f"experiment={self.experiment_name})"
        )
