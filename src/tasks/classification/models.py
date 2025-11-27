"""Classification models implementation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.core.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier implementation.

    A wrapper around sklearn's RandomForestClassifier that follows
    the BaseModel interface.

    Attributes:
        task_type: Always 'classification' for this model.
        params: Hyperparameters for the RandomForest.
        model: The underlying sklearn model instance.
    """

    task_type = "classification"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the Random Forest model.

        Args:
            params: Hyperparameters passed to RandomForestClassifier.
                Common params: n_estimators, max_depth, random_state, etc.
        """
        super().__init__(params)
        default_params = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.params = {**default_params, **self.params}

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Train the Random Forest classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Raises:
            ValueError: If y is None (required for classification).
        """
        if y is None:
            raise ValueError("Target labels (y) are required for classification.")

        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities for each class.

        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)  # type: ignore[no-any-return]

    @property
    def feature_importances(self) -> np.ndarray | None:
        """Get feature importances from the trained model.

        Returns:
            Array of feature importances or None if not trained.
        """
        if self.model is None:
            return None
        return self.model.feature_importances_  # type: ignore[no-any-return]


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier implementation.

    A wrapper around sklearn's LogisticRegression that follows
    the BaseModel interface.

    Attributes:
        task_type: Always 'classification' for this model.
        params: Hyperparameters for the LogisticRegression.
        model: The underlying sklearn model instance.
    """

    task_type = "classification"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the Logistic Regression model.

        Args:
            params: Hyperparameters passed to LogisticRegression.
                Common params: C, max_iter, random_state, solver, etc.
        """
        super().__init__(params)
        default_params = {
            "random_state": 42,
            "max_iter": 1000,
            "n_jobs": -1,
        }
        self.params = {**default_params, **self.params}

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Train the Logistic Regression classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Raises:
            ValueError: If y is None (required for classification).
        """
        if y is None:
            raise ValueError("Target labels (y) are required for classification.")

        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities for each class.

        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)  # type: ignore[no-any-return]
