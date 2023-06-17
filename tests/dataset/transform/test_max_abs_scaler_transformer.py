"""Test suite for `dataset.transform.max_abs_scaler_transformer`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.transform.max_abs_scaler_transformer import MaxAbsScalerTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[1, 2, 3, 4, 5],
        b=[-1, -2, 0, 1, 2],
    )


class TestMaxAbsScalerTransformer:
    """Test suite for `dataset.transform.max_abs_scaler_transformer.MaxAbsScalerTransformer`."""

    def test_label_encoding(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly scale the specified features."""
        max_abs_scaler_transformer = MaxAbsScalerTransformer(temporary_directory, features=["a", "b"])
        df = max_abs_scaler_transformer._transform(example_vaex_dataframe)

        assert df["a"].tolist() == [0.2, 0.4, 0.6, 0.8, 1.0]  # type: ignore
        assert df["b"].tolist() == [-0.5, -1.0, 0.0, 0.5, 1.0]  # type: ignore

    def test_cache(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly scale the specified features and use cache if possible."""
        MaxAbsScalerTransformer(temporary_directory, features=["a", "b"]).transform(example_vaex_dataframe)

        with patch.object(MaxAbsScalerTransformer, "_transform") as mocked_transform:
            MaxAbsScalerTransformer(temporary_directory, features=["a", "b"]).transform(example_vaex_dataframe)
            mocked_transform.assert_not_called()
