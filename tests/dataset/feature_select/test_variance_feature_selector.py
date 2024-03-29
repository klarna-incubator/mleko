"""Test suite for `dataset.feature_select.variance_feature_selector`."""

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.feature_select.variance_feature_selector import VarianceFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    m = 1000
    a_col = [100, 100, 101, 100, 100]
    a_m_col = [a * m for a in a_col]

    df = vaex.from_arrays(
        a=a_col,
        a_m=a_m_col,
        b=[1, 1, 1, 1, 1],
        string_col=["a", "b", "c", "d", "e"],
        target=[0, 0, 1, 0, 1],
    )
    return df


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(numerical=["a", "a_m", "b", "target"], categorical=["string_col"])


class TestVarianceFeatureSelector:
    """Test suite for `dataset.feature_select.variance_feature_selector.VarianceFeatureSelector`."""

    def test_var_invariant(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should drop columns with standard deviation of 0."""
        variance_feature_selector = VarianceFeatureSelector(
            cache_directory=temporary_directory, ignore_features=["target"], variance_threshold=0
        )
        (_, _, df) = variance_feature_selector._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df.shape == (5, 4)
        assert df.column_names == ["a", "a_m", "string_col", "target"]

    def test_var_different_scales(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should drop columns depending on normalized standard deviation threshold."""
        variance_feature_selector = VarianceFeatureSelector(
            cache_directory=temporary_directory, ignore_features=["b", "target"], variance_threshold=0.004
        )
        (_, _, df) = variance_feature_selector._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df.shape == (5, 3)
        assert df.column_names == ["b", "string_col", "target"]

    def test_var_default(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should perform standard deviation feature selection on numeric columns by default."""
        (_, _, df) = VarianceFeatureSelector(
            cache_directory=temporary_directory, variance_threshold=0.004
        ).fit_transform(example_data_schema, example_vaex_dataframe)
        assert df.shape == (5, 2)
        assert df.column_names == ["string_col", "target"]

        with patch.object(VarianceFeatureSelector, "_fit_transform") as mocked_fit_transform:
            VarianceFeatureSelector(cache_directory=temporary_directory, variance_threshold=0.004).fit_transform(
                example_data_schema, example_vaex_dataframe
            )
            mocked_fit_transform.assert_not_called()
