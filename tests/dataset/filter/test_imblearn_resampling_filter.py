"""Test suite for the `dataset.filter.imblearn_resampling_filter` module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest
import vaex
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.filter.expression_filter import ExpressionFilter
from mleko.dataset.filter.imblearn_resampling_filter import ImblearnResamplingFilter


@pytest.fixture(scope="module")
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=range(10),
        b=pa.array(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            mask=[True, True, True, True, True, True, True, True, True, True],
        ),
        target=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    )
    return df


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(numerical=["a", "b", "target"])


class TestExpressionFilter:
    """Test suite for `dataset.filter.imblearn_resampling_filter.ImblearnResamplingFilter`."""

    def test_under_sample(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should under-sample the dataframe using the `RandomUnderSampler`."""
        resampler = ImblearnResamplingFilter(
            RandomUnderSampler(), target_column="target", cache_directory=temporary_directory
        )

        df = resampler._filter(example_data_schema, example_vaex_dataframe)
        assert df.shape == (8, 3)
        assert df.get_column_names() == ["a", "b", "target"]
        assert df["target"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]  # type: ignore
        assert df["a"].tolist() == [0, 1, 2, 5, 6, 7, 8, 9]  # type: ignore

    def test_over_sample(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should over-sample the dataframe using the `RandomOverSampler`."""
        resampler = ImblearnResamplingFilter(
            RandomOverSampler(), target_column="target", cache_directory=temporary_directory
        )

        df = resampler._filter(example_data_schema, example_vaex_dataframe)
        assert df.shape == (12, 4)
        assert df.get_column_names() == ["a", "b", "target", "Synthetic"]
        assert df["target"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]  # type: ignore
        assert df["a"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8]  # type: ignore
        assert df["Synthetic"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # type: ignore

    def test_mocked_filter_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should test the cache of the imblearn resampling filter."""
        resampler = ImblearnResamplingFilter(
            RandomOverSampler(), target_column="target", cache_directory=temporary_directory
        )

        resampler.filter(example_data_schema, example_vaex_dataframe)
        with patch.object(ExpressionFilter, "_filter") as mocked_filter:
            resampler.filter(example_data_schema, example_vaex_dataframe)
            mocked_filter.assert_not_called()
