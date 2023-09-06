"""Test suite for `dataset.transform.min_max_scaler_transformer`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.transform.min_max_scaler_transformer import MinMaxScalerTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[1, 2, 3, 4, 5],
        b=[-1, -2, 0, 1, 2],
    )


class TestMinMaxScalerTransformer:
    """Test suite for `dataset.transform.min_max_scaler_transformer.MinMaxScalerTransformer`."""

    def test_min_max_scaling_default_range(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly scale the specified features."""
        min_max_scaler_transformer = MinMaxScalerTransformer(temporary_directory, features=["a", "b"])
        _, df = min_max_scaler_transformer._fit_transform(example_vaex_dataframe)

        assert df["a"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]  # type: ignore
        assert df["b"].tolist() == [0.25, 0.0, 0.5, 0.75, 1.0]  # type: ignore

    def test_min_max_scaling_custom_range(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly scale the specified features on a larger range."""
        min_max_scaler_transformer = MinMaxScalerTransformer(
            temporary_directory, features=["a", "b"], min_value=-1, max_value=1
        )
        _, df = min_max_scaler_transformer._fit_transform(example_vaex_dataframe)

        assert df["a"].tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]  # type: ignore
        assert df["b"].tolist() == [-0.5, -1.0, 0.0, 0.5, 1.0]  # type: ignore

    def test_cache(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly scale the specified features and use cache if possible."""
        MinMaxScalerTransformer(temporary_directory, features=["a", "b"]).fit_transform(example_vaex_dataframe)

        with patch.object(MinMaxScalerTransformer, "_fit_transform") as mocked_fit_transform:
            MinMaxScalerTransformer(temporary_directory, features=["a", "b"]).fit_transform(example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()
