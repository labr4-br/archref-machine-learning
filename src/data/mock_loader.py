"""Mock data loader for testing and development."""

from typing import Any

import numpy as np
import pandas as pd

from src.core.base_loader import BaseDataLoader


class MockDataLoader(BaseDataLoader):
    """Mock data loader that generates synthetic data.

    Useful for testing the pipeline without real data.
    Generates random features and binary classification targets.

    Attributes:
        n_samples: Number of samples to generate.
        n_features: Number of features to generate.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        n_samples: int = 1000,
        n_features: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize the mock data loader.

        Args:
            config: Configuration dictionary (optional).
            n_samples: Number of samples to generate.
            n_features: Number of features to generate.
            random_state: Random seed for reproducibility.
        """
        super().__init__(config)
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def load_raw(self) -> pd.DataFrame:
        """Generate synthetic classification data.

        Returns:
            DataFrame with random features and binary target.
        """
        np.random.seed(self.random_state)

        # Generate random features
        X = np.random.rand(self.n_samples, self.n_features)

        # Generate binary target with some signal
        # Create target based on sum of first two features
        signal = X[:, 0] + X[:, 1]
        threshold = np.median(signal)
        y = (signal > threshold).astype(int)

        # Add some noise to make it more realistic
        noise_idx = np.random.choice(
            self.n_samples, size=int(self.n_samples * 0.1), replace=False
        )
        y[noise_idx] = 1 - y[noise_idx]

        # Create DataFrame
        feature_names = [f"f{i}" for i in range(self.n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the mock data.

        For mock data, no processing is needed.

        Args:
            df: Raw DataFrame.

        Returns:
            Same DataFrame (no processing applied).
        """
        return df
