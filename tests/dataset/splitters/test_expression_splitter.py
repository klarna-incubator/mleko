"""Test suite for the `dataset.splitters.expression_splitter` module."""
from __future__ import annotations

from pathlib import Path

import pytest
import vaex
from mleko.dataset.splitters.expression_splitter import ExpressionSplitter


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


class TestExpressionDataSplitter:
    """Test suite for `dataset.splitters.expression_splitter.ExpressionDataSplitter`."""

    def test_split_by_index(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes based on index."""
        test_expression_data_splitter = ExpressionSplitter(temporary_directory, expression="(a < 2) | (a > 7)")

        df_train, df_test = test_expression_data_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (4, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 1, 0]  # type: ignore
        assert df_train["a"].tolist() == [0, 1, 8, 9]  # type: ignore
        assert df_test.shape == (6, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [0, 0, 0, 1, 1, 1]  # type: ignore

    def test_split_by_date(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes based on date."""
        test_expression_data_splitter = ExpressionSplitter(
            temporary_directory, expression='date < scalar_datetime("2020-06-01 00:00:00")'
        )

        df_train, df_test = test_expression_data_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 0, 0]  # type: ignore
        assert df_train["b"].tolist() == [  # type: ignore
            "2020-01-01 00:00:00",
            "2020-02-01 00:00:00",
            "2020-03-01 00:00:00",
            "2020-04-01 00:00:00",
            "2020-05-01 00:00:00",
        ]
        assert df_test.shape == (5, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [1, 1, 1, 1, 0]  # type: ignore
