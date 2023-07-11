"""Test suite for `dataset.feature_select.composite_feature_selector`."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import vaex

from mleko.dataset.feature_select.composite_feature_selector import CompositeFeatureSelector
from mleko.dataset.feature_select.missing_rate_feature_selector import MissingRateFeatureSelector
from mleko.dataset.feature_select.variance_feature_selector import VarianceFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=range(10),
        b=[
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        target=[0, 0, None, None, None, None, None, None, None, None],
    )
    return df


class TestCompositeFeatureSelector:
    """Test suite for `dataset.feature_select.composite_feature_selector.CompositeFeatureSelector`."""

    def test_select_chained_missing_std_cached(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should return vaex dataframe from feature_select method."""
        test_composite_feature_selector = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        )

        df_train = test_composite_feature_selector.select_features(example_vaex_dataframe, fit=True)
        assert df_train.shape == (10, 1)
        assert df_train.column_names == ["a"]

        with patch.object(CompositeFeatureSelector, "_select_features") as mocked_select_features:
            test_composite_feature_selector.select_features(example_vaex_dataframe, fit=False)
            mocked_select_features.assert_not_called()

    def test_persistent_feature_selector_loaded_from_disk(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should fit and transform the data and save the feature selector to disk."""
        df = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        ).select_features(example_vaex_dataframe, fit=True)
        first_cache = list(temporary_directory.glob("*"))

        assert df.shape == (10, 1)
        assert df.column_names == ["a"]

        df = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        ).select_features(example_vaex_dataframe, fit=True)
        second_cache = list(temporary_directory.glob("*"))

        assert df.shape == (10, 1)
        assert df.column_names == ["a"]
        assert first_cache == second_cache
