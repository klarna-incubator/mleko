"""Test suite for the `data.splitters` module."""
from __future__ import annotations

from pathlib import Path

import pytest
import vaex

from mleko.data.splitters import BaseDataSplitter, ExpressionDataSplitter, RandomDataSplitter


@pytest.fixture
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


class TestBaseDataSplitter:
    """Test suite for `data.splitters.BaseDataSplitter`."""

    class DerivedDataSplitter(BaseDataSplitter):
        """Test class."""

        def split(self, _file_paths):
            """Split."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6]), vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from convert method."""
        test_derived_data_splitter = self.DerivedDataSplitter(temporary_directory)

        df_train, df_test = test_derived_data_splitter.split([])
        assert df_train.shape == (3, 2)
        assert df_train.column_names == ["a", "b"]
        assert df_test.shape == (3, 2)
        assert df_test.column_names == ["a", "b"]


class TestRandomDataSplitter:
    """Test suite for `data.splitters.RandomDataSplitter`."""

    def test_split_shuffle_stratify(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes with shuffling and stratification."""
        test_random_data_splitter = RandomDataSplitter(
            temporary_directory, data_split=(0.5, 0.5), shuffle=True, stratify="target", random_state=1337
        )

        df_train, df_test = test_random_data_splitter.split(example_vaex_dataframe)
        assert df_train.shape == (5, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 1, 1]  # type: ignore
        assert df_test.shape == (5, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [0, 0, 1, 1, 0]  # type: ignore

    def test_split(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes without shuffling and stratification."""
        test_random_data_splitter = RandomDataSplitter(
            temporary_directory, data_split=(0.5, 0.5), shuffle=False, random_state=1337
        )

        df_train, df_test = test_random_data_splitter.split(example_vaex_dataframe)
        assert df_train.shape == (5, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 0, 0]  # type: ignore
        assert df_test.shape == (5, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [1, 1, 1, 1, 0]  # type: ignore


class TestExpressionDataSplitter:
    """Test suite for `data.splitters.ExpressionDataSplitter`."""

    def test_split_by_index(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes based on index."""
        test_expression_data_splitter = ExpressionDataSplitter(temporary_directory, expression="(a < 2) | (a > 7)")

        df_train, df_test = test_expression_data_splitter.split(example_vaex_dataframe)
        assert df_train.shape == (4, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 1, 0]  # type: ignore
        assert df_train["a"].tolist() == [0, 1, 8, 9]  # type: ignore
        assert df_test.shape == (6, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [0, 0, 0, 1, 1, 1]  # type: ignore

    def test_split_by_date(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes based on date."""
        test_expression_data_splitter = ExpressionDataSplitter(
            temporary_directory, expression='date < scalar_datetime("2020-06-01 00:00:00")'
        )

        df_train, df_test = test_expression_data_splitter.split(example_vaex_dataframe)
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
