"""Test suite for `dataset.feature_select.base_feature_selector`."""
from __future__ import annotations

from pathlib import Path

import pytest
import vaex

from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector


class TestBaseFeatureSelector:
    """Test suite for `dataset.feature_select.base_feature_selector.BaseFeatureSelector`."""

    class DerivedFeatureSelector(BaseFeatureSelector):
        """Test class."""

        def select_features(self, _dataframe):
            """Select Features."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

        def _default_features(self, dataframe):
            """Return default features."""
            return dataframe.get_column_names()

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from feature_select method."""
        test_derived_feature_selector = self.DerivedFeatureSelector(temporary_directory, [], None)

        df_train = test_derived_feature_selector.select_features([])
        assert df_train.shape == (3, 2)
        assert df_train.column_names == ["a", "b"]

    def test_mutually_exclusive_arguments(self, temporary_directory: Path):
        """Should raise ValueError when both `features` and `exclude_features` are provided."""
        with pytest.raises(ValueError):
            self.DerivedFeatureSelector(temporary_directory, [], [])
