"""Task-specific plugins for different ML task types."""

from typing import Literal

TaskType = Literal["classification", "regression", "clustering", "anomaly", "timeseries"]

__all__ = ["TaskType"]
