"""Tests for API endpoints."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_message(self, api_client: TestClient) -> None:
        """Test that root endpoint returns welcome message."""
        response = api_client.get("/")

        assert response.status_code == 200
        assert "message" in response.json()
        assert "API" in response.json()["message"]


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_prediction(
        self,
        api_client: TestClient,
        mock_model: MagicMock,
    ) -> None:
        """Test successful prediction."""
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        response = api_client.post(
            "/predict",
            json={"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "f4": 5.0},
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] == 1

    def test_predict_with_class_zero(
        self,
        api_client: TestClient,
        mock_model: MagicMock,
    ) -> None:
        """Test prediction returning class 0."""
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        response = api_client.post(
            "/predict",
            json={"f0": 0.1, "f1": 0.2, "f2": 0.3, "f3": 0.4, "f4": 0.5},
        )

        assert response.status_code == 200
        assert response.json()["prediction"] == 0

    def test_predict_without_model_returns_503(
        self,
        api_client_no_model: TestClient,
    ) -> None:
        """Test that prediction fails when model not loaded."""
        response = api_client_no_model.post(
            "/predict",
            json={"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "f4": 5.0},
        )

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_missing_feature(self, api_client: TestClient) -> None:
        """Test prediction with missing feature returns 422."""
        response = api_client.post(
            "/predict",
            json={"f0": 1.0, "f1": 2.0},  # Missing f2, f3, f4
        )

        assert response.status_code == 422

    def test_predict_invalid_type(self, api_client: TestClient) -> None:
        """Test prediction with invalid type returns 422."""
        response = api_client.post(
            "/predict",
            json={"f0": "invalid", "f1": 2.0, "f2": 3.0, "f3": 4.0, "f4": 5.0},
        )

        assert response.status_code == 422

    def test_predict_probability_format(
        self,
        api_client: TestClient,
        mock_model: MagicMock,
    ) -> None:
        """Test that probability is returned as list."""
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        response = api_client.post(
            "/predict",
            json={"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "f4": 5.0},
        )

        probability = response.json()["probability"]
        assert isinstance(probability, list)
        assert len(probability) == 2
        assert pytest.approx(sum(probability), 0.01) == 1.0

    def test_predict_model_without_proba(
        self,
        api_client_no_proba: TestClient,
    ) -> None:
        """Test prediction with model that doesn't support predict_proba."""
        response = api_client_no_proba.post(
            "/predict",
            json={"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "f4": 5.0},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["probability"] is None
