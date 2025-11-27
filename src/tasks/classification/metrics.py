"""Classification metrics implementation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.base_metrics import BaseMetrics


class ClassificationMetrics(BaseMetrics):
    """Metrics calculator for classification tasks.

    Calculates standard classification metrics including accuracy, precision,
    recall, F1 score, and optionally ROC AUC for binary/multiclass problems.

    Attributes:
        average: Averaging strategy for multiclass ('micro', 'macro', 'weighted').
    """

    def __init__(self, average: str = "weighted") -> None:
        """Initialize the metrics calculator.

        Args:
            average: Averaging strategy for multiclass metrics.
        """
        self.average = average

    def calculate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate all classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional, for ROC AUC).
            **_kwargs: Additional arguments (unused, for interface compatibility).

        Returns:
            Dictionary with all metrics.
        """
        metrics: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=self.average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=self.average, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

        # Calculate ROC AUC if probabilities are provided
        if y_proba is not None:
            try:
                # Binary classification
                if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                    proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics["roc_auc"] = roc_auc_score(y_true, proba)
                # Multiclass
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=self.average
                    )
            except ValueError:
                # ROC AUC may fail for certain edge cases
                metrics["roc_auc"] = None

        return metrics

    def primary_metric_name(self) -> str:
        """Return the primary metric for classification.

        Returns:
            'f1' as the primary metric.
        """
        return "f1"

    def get_flat_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Get metrics in a flat format suitable for logging.

        This method returns only scalar metrics, excluding complex objects
        like confusion matrices and classification reports.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).

        Returns:
            Dictionary with scalar metrics only.
        """
        all_metrics = self.calculate(y_true, y_pred, y_proba)

        flat_metrics: dict[str, float] = {
            "accuracy": all_metrics["accuracy"],
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "f1": all_metrics["f1"],
        }

        if all_metrics.get("roc_auc") is not None:
            flat_metrics["roc_auc"] = all_metrics["roc_auc"]

        return flat_metrics
