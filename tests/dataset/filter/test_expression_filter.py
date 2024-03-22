"""Test suite for the `dataset.filter.expression_filter` module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.filter.expression_filter import ExpressionFilter


@pytest.fixture(scope="module")
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=range(10),
        b=[
            "2020-01-01 00:00:00",
            "2020-02-01 00:00:00",
            "2020-03-01 00:00:00",
            "2020-04-01 00:00:00",
            "2020-05-01 00:00:00",
            "2020-06-01 00:00:00",
            "2020-07-01 00:00:00",
            "2020-08-01 00:00:00",
            "2020-09-01 00:00:00",
            "2020-10-01 00:00:00",
        ],
        target=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    )
    df["date"] = df.b.astype("datetime64[ns]")
    return df


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(numerical=["a", "target"], datetime=["b"])


class TestExpressionFilter:
    """Test suite for `dataset.filter.expression_filter.ExpressionFilter`."""

    def test_filter_by_index(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should filter the dataframe based on index."""
        test_expression_filter = ExpressionFilter(cache_directory=temporary_directory, expression="(a < 2) | (a > 7)")

        df = test_expression_filter._filter(example_data_schema, example_vaex_dataframe)
        assert df.shape == (4, 4)
        assert df.column_names == ["a", "b", "target", "date"]
        assert df["target"].tolist() == [0, 0, 1, 0]  # type: ignore
        assert df["a"].tolist() == [0, 1, 8, 9]  # type: ignore

    def test_filter_by_date(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should filter the dataframe based on date."""
        test_expression_filter = ExpressionFilter(
            cache_directory=temporary_directory, expression='date < scalar_datetime("2020-06-01 00:00:00")'
        )

        df = test_expression_filter._filter(example_data_schema, example_vaex_dataframe)
        assert df.shape == (5, 4)
        assert df.column_names == ["a", "b", "target", "date"]
        assert df["target"].tolist() == [0, 0, 0, 0, 0]  # type: ignore
        assert df["b"].tolist() == [  # type: ignore
            "2020-01-01 00:00:00",
            "2020-02-01 00:00:00",
            "2020-03-01 00:00:00",
            "2020-04-01 00:00:00",
            "2020-05-01 00:00:00",
        ]

    def test_filter_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should test the cache of the expression filter."""
        test_expression_filter = ExpressionFilter(
            cache_directory=temporary_directory, expression='date < scalar_datetime("2020-06-01 00:00:00")'
        )
        test_expression_filter.filter(example_data_schema, example_vaex_dataframe)

        with patch.object(ExpressionFilter, "_filter") as mocked_filter:
            test_expression_filter.filter(example_data_schema, example_vaex_dataframe)
            mocked_filter.assert_not_called()
