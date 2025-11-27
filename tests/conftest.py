"""Pytest fixtures for ML ArchRef tests."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import AppConfig, get_default_config
from src.tasks.classification.models import RandomForestModel


@pytest.fixture
def sample_config() -> AppConfig:
    """Provide a default configuration for tests."""
    return get_default_config()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.rand(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


@pytest.fixture
def sample_features(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Provide sample features (X) for testing."""
    return sample_dataframe.drop("target", axis=1)


@pytest.fixture
def sample_target(sample_dataframe: pd.DataFrame) -> pd.Series:
    """Provide sample target (y) for testing."""
    return sample_dataframe["target"]


@pytest.fixture
def trained_model(
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
) -> RandomForestModel:
    """Provide a trained RandomForest model for testing."""
    model = RandomForestModel({"n_estimators": 10, "random_state": 42})
    model.train(sample_features, sample_target)
    return model


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_yaml_content() -> str:
    """Provide sample YAML config content."""
    return """
project:
  name: "Test Project"
  version: "0.1.0"

task:
  type: "classification"

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  models: "models"
  logs: "reports/logs/app.log"

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


@pytest.fixture
def config_file(temp_dir: Path, config_yaml_content: str) -> Path:
    """Create a temporary config file for testing."""
    config_path = temp_dir / "config.yaml"
    config_path.write_text(config_yaml_content)
    return config_path


# API Test Fixtures


@pytest.fixture
def mock_model() -> MagicMock:
    """Provide a mock model for API testing."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


def _create_test_app_with_model(model_instance: MagicMock | None):
    """Create a FastAPI app with injected model for testing."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as PydanticModel

    import pandas as pd

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        yield

    test_app = FastAPI(lifespan=test_lifespan)

    class PredictionRequest(PydanticModel):
        f0: float
        f1: float
        f2: float
        f3: float
        f4: float

    @test_app.get("/")
    def read_root():
        return {"message": "ML Classification API is online!"}

    @test_app.post("/predict")
    def predict(request: PredictionRequest):
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        data = pd.DataFrame([request.model_dump()])
        prediction = model_instance.predict(data)[0]

        probability = None
        if hasattr(model_instance, "predict_proba"):
            probability = model_instance.predict_proba(data)[0].tolist()

        return {"prediction": int(prediction), "probability": probability}

    return test_app


@pytest.fixture
def api_client(mock_model: MagicMock) -> Generator[TestClient, None, None]:
    """Provide a TestClient with mocked model."""
    test_app = _create_test_app_with_model(mock_model)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def api_client_no_model() -> Generator[TestClient, None, None]:
    """Provide a TestClient without loaded model."""
    test_app = _create_test_app_with_model(None)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def api_client_no_proba() -> Generator[TestClient, None, None]:
    """Provide a TestClient with model without predict_proba."""
    model = MagicMock(spec=["predict"])
    model.predict.return_value = np.array([0])

    test_app = _create_test_app_with_model(model)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client
