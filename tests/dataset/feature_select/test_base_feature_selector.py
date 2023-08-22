"""Test suite for `dataset.feature_select.base_feature_selector`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["1", "1", "0", "0"],
        b=["1", "1", "1", "1"],
        c=[None, "1", "1", "1"],
    )


class TestBaseFeatureSelector:
    """Test suite for `dataset.feature_select.base_feature_selector.BaseFeatureSelector`."""

    class DerivedFeatureSelector(BaseFeatureSelector):
        """Test class."""

        def _default_features(self, dataframe):
            """Return default features."""
            return dataframe.get_column_names()

        def _fingerprint(self):
            """Return fingerprint."""
            return "fingerprint"

        def _fit(self, _dataframe):
            """Fit feature selector."""
            return 1337

        def _transform(self, _dataframe):
            """Select features."""
            return _dataframe

    def test_fit_and_transform(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should fit and transform dataframe."""
        test_derived_transformer = self.DerivedFeatureSelector(temporary_directory, [], None, 1)

        transformer = test_derived_transformer.fit(example_vaex_dataframe)
        df = test_derived_transformer.transform(example_vaex_dataframe)
        assert transformer == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

        with patch.object(self.DerivedFeatureSelector, "_fit") as mocked_fit:
            test_derived_transformer.fit(example_vaex_dataframe)
            mocked_fit.assert_not_called()

        with patch.object(self.DerivedFeatureSelector, "_transform") as mocked_transform:
            test_derived_transformer.transform(example_vaex_dataframe)
            mocked_transform.assert_not_called()

    def test_abstract_methods(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should return vaex dataframe from feature_select method."""
        test_derived_feature_selector = self.DerivedFeatureSelector(temporary_directory, [], None, 1)

        feature_selector, df = test_derived_feature_selector.fit_transform(example_vaex_dataframe)
        assert feature_selector == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

    def test_error_on_transform_before_fit(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should raise error when transform is called before fit."""
        test_derived_feature_selector = self.DerivedFeatureSelector(temporary_directory, [], None, 1)

        with pytest.raises(RuntimeError):
            test_derived_feature_selector.transform(example_vaex_dataframe)

    def test_mutually_exclusive_arguments(self, temporary_directory: Path):
        """Should raise ValueError when both `features` and `exclude_features` are provided."""
        with pytest.raises(ValueError):
            self.DerivedFeatureSelector(temporary_directory, [], [], 1)
