"""Test suite for the `utils.vaex_helpers` module."""

from __future__ import annotations

import pytest
import vaex

from mleko.utils.vaex_helpers import HashableVaexDataFrame, get_column, get_columns, get_filtered_df, get_indices


@pytest.fixture(scope="module")
def example_vaex_dataframe():
    """Return an example vaex DataFrame."""
    data = {"column1": [1, 2, 3], "column2": [4, 5, 6], "column3": [7, 8, 9]}
    return vaex.from_dict(data)


class TestGetColumn:
    """Test suite for `utils.vaex_helpers.get_column`."""

    def test_get_column(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return an Expression containing only the specified column."""
        result = get_column(example_vaex_dataframe, "column1")
        assert result.expression == "column1"
        assert result.tolist() == [1, 2, 3]

    def test_get_nonexistent_column(self, example_vaex_dataframe: vaex.DataFrame):
        """Should raise a NameError if a non-existent column is specified."""
        with pytest.raises(NameError):
            get_column(example_vaex_dataframe, "column4")


class TestGetColumns:
    """Test suite for `utils.vaex_helpers.get_columns`."""

    def test_get_columns(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the specified columns."""
        result = get_columns(example_vaex_dataframe, ["column1", "column3"])
        assert result.get_column_names() == ["column1", "column3"]
        assert result.column_count() == 2
        assert result["column1"].tolist() == [1, 2, 3]  # type: ignore
        assert result["column3"].tolist() == [7, 8, 9]  # type: ignore

    def test_get_nonexistent_column(self, example_vaex_dataframe: vaex.DataFrame):
        """Should raise a NameError if a non-existent column is specified."""
        with pytest.raises(NameError):
            get_columns(example_vaex_dataframe, ["column1", "column4"])


class TestGetFilteredDf:
    """Test suite for `utils.vaex_helpers.get_filtered_df`."""

    def test_get_filtered_df(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the rows that satisfy the filter."""
        result = get_filtered_df(example_vaex_dataframe, example_vaex_dataframe.column1 > 1)
        assert result.get_column_names() == ["column1", "column2", "column3"]
        assert result["column1"].tolist() == [2, 3]  # type: ignore
        assert result["column2"].tolist() == [5, 6]  # type: ignore
        assert result["column3"].tolist() == [8, 9]  # type: ignore

    def test_no_matching_rows(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return an empty DataFrame if no rows match the filter."""
        result = get_filtered_df(example_vaex_dataframe, example_vaex_dataframe.column1 > 3)
        assert result.get_column_names() == ["column1", "column2", "column3"]
        assert result.length_original() == 3
        assert result.length() == 0
        assert result["column1"].tolist() == []  # type: ignore


class TestGetIndices:
    """Test suite for `utils.vaex_helpers.get_indices`."""

    def test_get_indices(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the specified indices."""
        result = get_indices(example_vaex_dataframe, [0, 2])
        assert result.length_original() == 2
        assert result.column_count() == 3
        assert result["column1"].tolist() == [1, 3]  # type: ignore
        assert result["column2"].tolist() == [4, 6]  # type: ignore
        assert result["column3"].tolist() == [7, 9]  # type: ignore

    def test_negative_indices(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the specified indices."""
        result = get_indices(example_vaex_dataframe, [-1, -2])
        assert result.length_original() == 0
        assert result.column_count() == 3
        assert result["column1"].tolist() == []  # type: ignore
        assert result["column2"].tolist() == []  # type: ignore
        assert result["column3"].tolist() == []  # type: ignore


class TestHashableVaexDataFrame:
    """Test suite for `utils.vaex_helpers.HashableVaexDataFrame`."""

    def test_hashable_vaex_dataframe(self, example_vaex_dataframe: vaex.DataFrame):
        """Identical DataFrames should have the same hash and be equal."""
        hashable_df1 = HashableVaexDataFrame(example_vaex_dataframe)
        hashable_df2 = HashableVaexDataFrame(example_vaex_dataframe)
        assert hashable_df1 == hashable_df2
        assert hash(hashable_df1) == hash(hashable_df2)

    def test_inequality(self):
        """Different DataFrames should have different hashes and not be equal."""
        df1 = vaex.from_arrays(column1=[1, 2, 3], column2=[4, 5, 6], column3=[7, 8, 9])
        df2 = vaex.from_arrays(column1=[1, 2, 3], column2=[4, 5, 6], column3=[7, 8, 10])
        hashable_df1 = HashableVaexDataFrame(df1)
        hashable_df2 = HashableVaexDataFrame(df2)
        assert hashable_df1 != hashable_df2
