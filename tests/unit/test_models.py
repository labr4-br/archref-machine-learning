"""Tests for model classes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tasks.classification.models import LogisticRegressionModel, RandomForestModel


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_init_default_params(self) -> None:
        """Test model initialization with default parameters."""
        model = RandomForestModel()
        assert model.task_type == "classification"
        assert model.params["n_estimators"] == 100
        assert model.params["random_state"] == 42
        assert model.is_trained is False

    def test_init_custom_params(self) -> None:
        """Test model initialization with custom parameters."""
        model = RandomForestModel({"n_estimators": 50, "max_depth": 5})
        assert model.params["n_estimators"] == 50
        assert model.params["max_depth"] == 5

    def test_train(
        self,
        sample_features: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test model training."""
        model = RandomForestModel({"n_estimators": 10})
        model.train(sample_features, sample_target)

        assert model.is_trained is True
        assert model.model is not None

    def test_train_requires_target(self, sample_features: pd.DataFrame) -> None:
        """Test that training requires target labels."""
        model = RandomForestModel()
        with pytest.raises(ValueError, match="Target labels"):
            model.train(sample_features, None)

    def test_predict(self, trained_model: RandomForestModel, sample_features: pd.DataFrame) -> None:
        """Test model prediction."""
        predictions = trained_model.predict(sample_features)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_features)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_untrained_raises(self, sample_features: pd.DataFrame) -> None:
        """Test that predicting with untrained model raises error."""
        model = RandomForestModel()
        with pytest.raises(ValueError, match="not trained"):
            model.predict(sample_features)

    def test_predict_proba(
        self,
        trained_model: RandomForestModel,
        sample_features: pd.DataFrame,
    ) -> None:
        """Test probability prediction."""
        probabilities = trained_model.predict_proba(sample_features)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(sample_features), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_save_and_load(
        self,
        trained_model: RandomForestModel,
        sample_features: pd.DataFrame,
        temp_dir: Path,
    ) -> None:
        """Test model save and load."""
        model_path = temp_dir / "model.pkl"

        # Save
        trained_model.save(model_path)
        assert model_path.exists()

        # Load into new instance
        new_model = RandomForestModel()
        new_model.load(model_path)

        # Verify predictions match
        original_preds = trained_model.predict(sample_features)
        loaded_preds = new_model.predict(sample_features)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_untrained_raises(self, temp_dir: Path) -> None:
        """Test that saving untrained model raises error."""
        model = RandomForestModel()
        with pytest.raises(ValueError, match="untrained"):
            model.save(temp_dir / "model.pkl")

    def test_load_nonexistent_raises(self) -> None:
        """Test that loading from nonexistent file raises error."""
        model = RandomForestModel()
        with pytest.raises(FileNotFoundError):
            model.load("nonexistent.pkl")

    def test_feature_importances(self, trained_model: RandomForestModel) -> None:
        """Test feature importances property."""
        importances = trained_model.feature_importances

        assert importances is not None
        assert len(importances) == 5  # 5 features
        assert all(i >= 0 for i in importances)

    def test_repr(self) -> None:
        """Test string representation."""
        model = RandomForestModel()
        repr_str = repr(model)

        assert "RandomForestModel" in repr_str
        assert "classification" in repr_str


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_init(self) -> None:
        """Test model initialization."""
        model = LogisticRegressionModel()
        assert model.task_type == "classification"
        assert model.params["max_iter"] == 1000

    def test_train_and_predict(
        self,
        sample_features: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test training and prediction."""
        model = LogisticRegressionModel()
        model.train(sample_features, sample_target)

        predictions = model.predict(sample_features)
        assert len(predictions) == len(sample_features)
