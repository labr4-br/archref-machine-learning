"""Tests for metrics module."""

import numpy as np
import pytest

from src.tasks.classification.metrics import ClassificationMetrics


class TestClassificationMetrics:
    """Tests for ClassificationMetrics."""

    @pytest.fixture
    def metrics(self) -> ClassificationMetrics:
        """Provide metrics calculator instance."""
        return ClassificationMetrics()

    @pytest.fixture
    def perfect_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Provide perfect predictions (100% accuracy)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        return y_true, y_pred

    @pytest.fixture
    def imperfect_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Provide imperfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        return y_true, y_pred

    def test_calculate_perfect_predictions(
        self,
        metrics: ClassificationMetrics,
        perfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test metrics calculation with perfect predictions."""
        y_true, y_pred = perfect_predictions
        result = metrics.calculate(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_calculate_imperfect_predictions(
        self,
        metrics: ClassificationMetrics,
        imperfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test metrics calculation with imperfect predictions."""
        y_true, y_pred = imperfect_predictions
        result = metrics.calculate(y_true, y_pred)

        assert 0 < result["accuracy"] < 1
        assert "confusion_matrix" in result
        assert "classification_report" in result

    def test_calculate_with_probabilities(
        self,
        metrics: ClassificationMetrics,
        perfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test metrics calculation with probability scores."""
        y_true, y_pred = perfect_predictions
        y_proba = np.column_stack([1 - y_pred, y_pred]).astype(float)

        result = metrics.calculate(y_true, y_pred, y_proba=y_proba)

        assert "roc_auc" in result
        assert result["roc_auc"] == 1.0

    def test_primary_metric_name(self, metrics: ClassificationMetrics) -> None:
        """Test primary metric name."""
        assert metrics.primary_metric_name() == "f1"

    def test_primary_metric_value(
        self,
        metrics: ClassificationMetrics,
        perfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test primary metric value calculation."""
        y_true, y_pred = perfect_predictions
        value = metrics.primary_metric_value(y_true, y_pred)

        assert value == 1.0

    def test_get_flat_metrics(
        self,
        metrics: ClassificationMetrics,
        imperfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test flat metrics output."""
        y_true, y_pred = imperfect_predictions
        flat = metrics.get_flat_metrics(y_true, y_pred)

        assert "accuracy" in flat
        assert "precision" in flat
        assert "recall" in flat
        assert "f1" in flat
        # Should not include complex objects
        assert "confusion_matrix" not in flat
        assert "classification_report" not in flat

    def test_repr(self, metrics: ClassificationMetrics) -> None:
        """Test string representation."""
        repr_str = repr(metrics)
        assert "ClassificationMetrics" in repr_str
        assert "f1" in repr_str

    def test_custom_average(self) -> None:
        """Test metrics with custom averaging strategy."""
        metrics = ClassificationMetrics(average="macro")

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])

        result = metrics.calculate(y_true, y_pred)
        assert "f1" in result
