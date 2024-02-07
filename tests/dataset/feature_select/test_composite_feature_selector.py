"""Test suite for `dataset.feature_select.composite_feature_selector`."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
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


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(numerical=["a", "b", "target"])


class TestCompositeFeatureSelector:
    """Test suite for `dataset.feature_select.composite_feature_selector.CompositeFeatureSelector`."""

    def test_select_chained_missing_std_cached(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe from feature_select method."""
        test_composite_feature_selector = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        )

        (_, _, df_train) = test_composite_feature_selector.fit_transform(example_data_schema, example_vaex_dataframe)
        assert df_train.shape == (10, 1)
        assert df_train.column_names == ["a"]

        with patch.object(CompositeFeatureSelector, "_fit_transform") as mocked_fit_transform:
            test_composite_feature_selector.fit_transform(example_data_schema, example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()

    def test_select_chained_missing_std_cached_separate_fit_transform(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe from feature_select method using separate fit and transform calls."""
        test_composite_feature_selector = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        )

        _, _ = test_composite_feature_selector.fit(example_data_schema, example_vaex_dataframe)
        _, df_train = test_composite_feature_selector.transform(example_data_schema, example_vaex_dataframe)
        assert df_train.shape == (10, 1)
        assert df_train.column_names == ["a"]

        with patch.object(CompositeFeatureSelector, "_transform") as mocked_transform:
            test_composite_feature_selector.transform(example_data_schema, example_vaex_dataframe)
            mocked_transform.assert_not_called()

    def test_persistent_feature_selector_loaded_from_disk(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should fit and transform the data and save the feature selector to disk."""
        (_, _, df) = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        ).fit_transform(example_data_schema, example_vaex_dataframe)
        first_cache = list(temporary_directory.glob("*"))

        assert df.shape == (10, 1)
        assert df.column_names == ["a"]

        (_, _, df) = CompositeFeatureSelector(
            temporary_directory,
            [
                MissingRateFeatureSelector(temporary_directory, missing_rate_threshold=0.5),
                VarianceFeatureSelector(temporary_directory, variance_threshold=0.0),
            ],
        ).fit_transform(example_data_schema, example_vaex_dataframe)
        second_cache = list(temporary_directory.glob("*"))

        assert df.shape == (10, 1)
        assert df.column_names == ["a"]
        assert first_cache == second_cache
