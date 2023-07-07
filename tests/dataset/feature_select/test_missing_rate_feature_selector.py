"""Test suite for `dataset.feature_select.missing_rate_feature_selector`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.feature_select.missing_rate_feature_selector import MissingRateFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=range(10),
        b=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            None,
            None,
            None,
        ],
        target=[0, 0, None, None, None, None, None, None, None, None],
    )
    return df


class TestMissingRateFeatureSelector:
    """Test suite for `dataset.feature_select.missing_rate_feature_selector.MissingRateFeatureSelector`."""

    def test_filter_missing_with_ignore(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should filter away columns with missing rate above threshold excluding ignored columns."""
        test_missing_rate_feature_selector = MissingRateFeatureSelector(
            temporary_directory, ignore_features=["target"], missing_rate_threshold=0.2
        )

        df = test_missing_rate_feature_selector._select_features(example_vaex_dataframe, fit=True)
        assert df.shape == (10, 2)
        assert df.column_names == ["a", "target"]

    def test_filter_missing_with_features(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should filter away columns with missing rate above threshold including only specified columns."""
        test_missing_rate_feature_selector = MissingRateFeatureSelector(
            temporary_directory, features=["b"], missing_rate_threshold=0.5
        )

        df = test_missing_rate_feature_selector._select_features(example_vaex_dataframe, fit=True)
        assert df.shape == (10, 3)
        assert df.column_names == ["a", "b", "target"]

    def test_filter_missing_cached(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should filter away columns with missing rate above threshold."""
        df = MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.2).select_features(
            example_vaex_dataframe, fit=True
        )
        assert df.shape == (10, 1)
        assert df.column_names == ["a"]

        with patch.object(MissingRateFeatureSelector, "_select_features") as mock_select_features:
            MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.2).select_features(
                example_vaex_dataframe, fit=False
            )
            mock_select_features.assert_not_called()
