"""FastAPI application for ML model serving."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

# Global variables
model: Any = None
config: dict[str, Any] | None = None
logger: logging.Logger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    global model, config, logger
    try:
        config = load_config("config.yaml")
        logger = setup_logger("api", config["paths"]["logs"])

        model_path = Path(config["paths"]["models"]) / f"{config['model']['name']}.pkl"

        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(
                f"Model not found at {model_path}. API will not be able to predict."
            )

    except Exception as e:
        print(f"Error during startup: {e}")

    yield
    # Shutdown
    if logger is not None:
        logger.info("Shutting down API")


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    f0: float
    f1: float
    f2: float
    f3: float
    f4: float


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint returning API status."""
    return {"message": "ML Classification API is online!"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, int | list[float] | None]:
    """Predict endpoint for classification.

    Args:
        request: Features for prediction.

    Returns:
        Dictionary with prediction and optional probabilities.

    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    global model, logger
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Make prediction
        prediction = model.predict(data)[0]

        # Get probability if available (RandomForest has it)
        probability: list[float] | None = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data)[0].tolist()

        return {"prediction": int(prediction), "probability": probability}
    except Exception as e:
        if logger is not None:
            logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
