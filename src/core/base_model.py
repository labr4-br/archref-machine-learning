"""Abstract base class for all ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

TaskType = Literal["classification", "regression", "clustering", "anomaly", "timeseries"]


class BaseModel(ABC):
    """Abstract base class for all Machine Learning models.

    This class defines the standard interface that all models must implement.
    It provides common functionality for saving/loading models while requiring
    subclasses to implement task-specific training and prediction logic.

    Attributes:
        task_type: The type of ML task this model performs.
        params: Configuration parameters for the model.
        model: The underlying model instance (set after training).
    """

    task_type: TaskType
    params: dict[str, Any]
    model: Any | None = None

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the model with configuration parameters.

        Args:
            params: Dictionary of model hyperparameters.
        """
        self.params = params or {}
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Train the model on the provided data.

        Args:
            X: Feature matrix.
            y: Target vector. Optional for unsupervised tasks.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions on new data.

        Args:
            X: Feature matrix.

        Returns:
            Model predictions (format depends on task type).
        """
        pass

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            path: File path to save the model.

        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to load the model from.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.model is not None

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"{self.__class__.__name__}(task_type={self.task_type}, trained={self.is_trained})"
