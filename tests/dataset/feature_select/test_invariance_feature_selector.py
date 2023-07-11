"""Test suite for `dataset.feature_select.invariance_feature_selector`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.feature_select.invariance_feature_selector import InvarianceFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[1, 1, 1, 1, 1],
        b=["1", "1", "1", "1", "1"],
        c=[True, True, True, True, True],
        d=[False, False, False, False, True],
        e=[None, "1", "1", "1", "1"],
    )


class TestInvarianceFeatureSelector:
    """Test suite for `dataset.feature_select.invariance_feature_selector.InvarianceFeatureSelector`."""

    def test_default_invariant(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop invariant categorical and boolean columns."""
        invariance_feature_selector = InvarianceFeatureSelector(temporary_directory)
        df = invariance_feature_selector._select_features(example_vaex_dataframe, fit=True)
        assert df.shape == (5, 3)
        assert df.column_names == ["a", "d", "e"]

    def test_invariant_numeric(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should include numerical columns if specified."""
        invariance_feature_selector = InvarianceFeatureSelector(temporary_directory, features=["a", "b", "d", "e"])
        df = invariance_feature_selector._select_features(example_vaex_dataframe, fit=True)
        assert df.shape == (5, 3)
        assert df.column_names == ["c", "d", "e"]

    def test_cache(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly drop invariant columns and use cache if possible."""
        df = InvarianceFeatureSelector(temporary_directory).select_features(example_vaex_dataframe, fit=True)
        assert df.shape == (5, 3)
        assert df.column_names == ["a", "d", "e"]

        with patch.object(InvarianceFeatureSelector, "_select_features") as mocked_select_features:
            InvarianceFeatureSelector(temporary_directory).select_features(example_vaex_dataframe, fit=False)
            mocked_select_features.assert_not_called()
