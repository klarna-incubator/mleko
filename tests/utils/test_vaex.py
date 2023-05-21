"""Test suite for the `utils.vaex` module."""
from __future__ import annotations

import pytest
import vaex

from mleko.utils.vaex import get_column, get_columns, get_indices


@pytest.fixture
def example_vaex_dataframe():
    """Return an example vaex DataFrame."""
    data = {"column1": [1, 2, 3], "column2": [4, 5, 6], "column3": [7, 8, 9]}
    return vaex.from_dict(data)


class TestGetColumn:
    """Test suite for `utils.vaex.get_column`."""

    def test_get_column(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return an Expression containing only the specified column."""
        result = get_column(example_vaex_dataframe, "column1")
        assert result.expression == "column1"
        assert result.tolist() == [1, 2, 3]


class TestGetColumns:
    """Test suite for `utils.vaex.get_columns`."""

    def test_get_columns(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the specified columns."""
        result = get_columns(example_vaex_dataframe, ["column1", "column3"])
        assert result.get_column_names() == ["column1", "column3"]
        assert result.column_count() == 2
        assert result["column1"].tolist() == [1, 2, 3]  # type: ignore
        assert result["column3"].tolist() == [7, 8, 9]  # type: ignore


class TestGetIndices:
    """Test suite for `utils.vaex.get_indices`."""

    def test_get_indices(self, example_vaex_dataframe: vaex.DataFrame):
        """Should return a DataFrame containing only the specified indices."""
        result = get_indices(example_vaex_dataframe, [0, 2])
        assert result.length_original() == 2
        assert result.column_count() == 3
        assert result["column1"].tolist() == [1, 3]  # type: ignore
        assert result["column2"].tolist() == [4, 6]  # type: ignore
        assert result["column3"].tolist() == [7, 9]  # type: ignore
