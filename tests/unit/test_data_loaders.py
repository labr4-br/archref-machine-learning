"""Tests for data loader module."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.mock_loader import MockDataLoader


class TestMockDataLoader:
    """Tests for MockDataLoader."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        loader = MockDataLoader()

        assert loader.n_samples == 1000
        assert loader.n_features == 5
        assert loader.random_state == 42

    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        loader = MockDataLoader(
            n_samples=500,
            n_features=10,
            random_state=123,
        )

        assert loader.n_samples == 500
        assert loader.n_features == 10
        assert loader.random_state == 123

    def test_load_raw(self) -> None:
        """Test raw data loading."""
        loader = MockDataLoader(n_samples=100, n_features=5)
        df = loader.load_raw()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 6  # 5 features + target
        assert "target" in df.columns

    def test_load_raw_features(self) -> None:
        """Test that features are properly named."""
        loader = MockDataLoader(n_features=3)
        df = loader.load_raw()

        expected_features = ["f0", "f1", "f2"]
        for feat in expected_features:
            assert feat in df.columns

    def test_load_raw_target_values(self) -> None:
        """Test that target contains binary values."""
        loader = MockDataLoader()
        df = loader.load_raw()

        assert df["target"].isin([0, 1]).all()

    def test_load_raw_reproducibility(self) -> None:
        """Test that loading is reproducible with same seed."""
        loader1 = MockDataLoader(random_state=42)
        loader2 = MockDataLoader(random_state=42)

        df1 = loader1.load_raw()
        df2 = loader2.load_raw()

        pd.testing.assert_frame_equal(df1, df2)

    def test_load_raw_different_seeds(self) -> None:
        """Test that different seeds produce different data."""
        loader1 = MockDataLoader(random_state=42)
        loader2 = MockDataLoader(random_state=123)

        df1 = loader1.load_raw()
        df2 = loader2.load_raw()

        # Should not be equal
        assert not df1.equals(df2)

    def test_process(self) -> None:
        """Test data processing."""
        loader = MockDataLoader()
        df = loader.load_raw()
        processed = loader.process(df)

        # For mock loader, process returns same data
        pd.testing.assert_frame_equal(df, processed)

    def test_load_and_process(self) -> None:
        """Test combined load and process."""
        loader = MockDataLoader(n_samples=50)
        df = loader.load_and_process()

        assert len(df) == 50
        assert "target" in df.columns

    def test_save_processed(self, temp_dir: Path) -> None:
        """Test saving processed data."""
        loader = MockDataLoader(n_samples=50)
        df = loader.load_raw()

        output_path = temp_dir / "processed.csv"
        loader.save_processed(df, output_path)

        assert output_path.exists()

        # Verify content
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 50

    def test_load_processed(self, temp_dir: Path) -> None:
        """Test loading processed data."""
        # Create a CSV file
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = temp_dir / "data.csv"
        df.to_csv(csv_path, index=False)

        loader = MockDataLoader()
        loaded = loader.load_processed(csv_path)

        pd.testing.assert_frame_equal(df, loaded)

    def test_load_processed_not_found(self) -> None:
        """Test error when loading nonexistent file."""
        loader = MockDataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_processed("nonexistent.csv")

    def test_repr(self) -> None:
        """Test string representation."""
        loader = MockDataLoader()
        repr_str = repr(loader)

        assert "MockDataLoader" in repr_str
