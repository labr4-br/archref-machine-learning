"""Abstract base class for metrics calculation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseMetrics(ABC):
    """Abstract base class for calculating model evaluation metrics.

    This class defines the standard interface for metrics calculation.
    Each task type (classification, regression, clustering) should have
    its own implementation with appropriate metrics.
    """

    @abstractmethod
    def calculate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate all relevant metrics for the task.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            **kwargs: Additional arguments specific to certain metrics.

        Returns:
            Dictionary with metric names and values.
        """
        pass

    @abstractmethod
    def primary_metric_name(self) -> str:
        """Return the name of the primary metric for this task.

        This is the metric that should be used for model selection
        and hyperparameter optimization.

        Returns:
            Name of the primary metric.
        """
        pass

    def primary_metric_value(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Calculate and return only the primary metric value.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            **kwargs: Additional arguments specific to certain metrics.

        Returns:
            Primary metric value.
        """
        metrics = self.calculate(y_true, y_pred, **kwargs)
        return float(metrics[self.primary_metric_name()])

    @abstractmethod
    def get_flat_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Get only scalar metrics suitable for logging.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            y_proba: Predicted probabilities (optional).

        Returns:
            Dictionary with only scalar metric values.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the metrics calculator."""
        return f"{self.__class__.__name__}(primary={self.primary_metric_name()})"
