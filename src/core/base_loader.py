"""Abstract base class for data loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseDataLoader(ABC):
    """Abstract base class for data loading and processing.

    This class defines the standard interface for loading and processing data.
    Subclasses must implement the load_raw() and process() methods for their
    specific data sources.

    Attributes:
        config: Configuration dictionary for the data loader.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the data loader with configuration.

        Args:
            config: Configuration dictionary with paths and settings.
        """
        self.config = config or {}

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Load raw data from source.

        Returns:
            DataFrame with raw data.
        """
        pass

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into features.

        Args:
            df: Raw DataFrame to process.

        Returns:
            Processed DataFrame ready for model training.
        """
        pass

    def load_and_process(self) -> pd.DataFrame:
        """Convenience method to load and process data in one step.

        Returns:
            Processed DataFrame ready for model training.
        """
        raw_df = self.load_raw()
        return self.process(raw_df)

    def save_processed(self, df: pd.DataFrame, path: str | Path) -> None:
        """Save processed data to disk.

        Args:
            df: DataFrame to save.
            path: File path for output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load_processed(self, path: str | Path) -> pd.DataFrame:
        """Load previously processed data from disk.

        Args:
            path: File path to load from.

        Returns:
            DataFrame with processed data.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Processed data file not found: {path}")

        return pd.read_csv(path)

    def __repr__(self) -> str:
        """Return string representation of the loader."""
        return f"{self.__class__.__name__}(config={list(self.config.keys())})"
