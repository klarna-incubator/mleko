"""Test suite for `dataset.feature_select.standard_deviation_feature_selector`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.feature_select.standard_deviation_feature_selector import StandardDeviationFeatureSelector


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


class TestStandardDeviationFeatureSelector:
    """Test suite for `dataset.feature_select.standard_deviation_feature_selector.StandardDeviationFeatureSelector`."""

    def test_std_invariant(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop columns with standard deviation of 0."""
        standard_deviation_feature_selector = StandardDeviationFeatureSelector(
            temporary_directory, ignore_features=["target"], standard_deviation_threshold=0
        )
        df = standard_deviation_feature_selector._select_features(example_vaex_dataframe)
        assert df.shape == (5, 4)
        assert df.column_names == ["a", "a_m", "string_col", "target"]

    def test_std_different_scales(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop columns depending on normalized standard deviation threshold."""
        standard_deviation_feature_selector = StandardDeviationFeatureSelector(
            temporary_directory, ignore_features=["b", "target"], standard_deviation_threshold=0.004
        )
        df = standard_deviation_feature_selector._select_features(example_vaex_dataframe)
        assert df.shape == (5, 3)
        assert df.column_names == ["b", "string_col", "target"]

    def test_std_default(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should perform standard deviation feature selection on numeric columns by default."""
        df = StandardDeviationFeatureSelector(temporary_directory, standard_deviation_threshold=0.004).select_features(
            example_vaex_dataframe
        )
        assert df.shape == (5, 2)
        assert df.column_names == ["string_col", "target"]

        with patch.object(StandardDeviationFeatureSelector, "_select_features") as mocked_select_features:
            StandardDeviationFeatureSelector(temporary_directory, standard_deviation_threshold=0.004).select_features(
                example_vaex_dataframe
            )
            mocked_select_features.assert_not_called()
