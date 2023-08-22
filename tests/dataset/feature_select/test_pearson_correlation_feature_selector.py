"""Test suite for `dataset.feature_select.pearson_correlation_feature_selector`."""
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.feature_select.pearson_correlation_feature_selector import PearsonCorrelationFeatureSelector


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[100, 100, 101, 100, 100],
        a_copy=[100, 100, 101, 100, 100],
        a_modified=[100, 100, 102, 100, 99.5],
        b=[1, 1, 1, 1, 130],
        string_col=["a", "b", "c", "d", "e"],
        target=[0, 0, 1, 0, 1],
    )


class TestPearsonCorrelationFeatureSelector:
    """Test suite for `dataset.feature_select.pearson_correlation_feature_selector.PearsonCorrelationFeatureSelector`."""  # noqa: E501

    def test_identical_features(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop identical columns."""
        correlation_feature_selector = PearsonCorrelationFeatureSelector(
            temporary_directory, ignore_features=["target"], correlation_threshold=1.0
        )
        _, df = correlation_feature_selector._fit_transform(example_vaex_dataframe)
        assert df.shape == (5, 5)
        assert df.column_names == ["a", "a_modified", "b", "string_col", "target"]

    def test_corr_resonable(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop features with correlation above threshold."""
        correlation_feature_selector = PearsonCorrelationFeatureSelector(
            temporary_directory, ignore_features=["b", "target"], correlation_threshold=0.7
        )
        _, df = correlation_feature_selector._fit_transform(example_vaex_dataframe)
        assert df.shape == (5, 4)
        assert df.column_names == ["a_modified", "b", "string_col", "target"]

    def test_corr_cached(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should cache the result of the feature selection."""
        _, df = PearsonCorrelationFeatureSelector(
            temporary_directory, ignore_features=["target"], correlation_threshold=0.7
        ).fit_transform(example_vaex_dataframe)
        assert df.shape == (5, 4)
        assert df.column_names == ["a", "b", "string_col", "target"]

        with patch.object(PearsonCorrelationFeatureSelector, "_fit_transform") as mocked_fit_transform:
            PearsonCorrelationFeatureSelector(
                temporary_directory, ignore_features=["target"], correlation_threshold=0.7
            ).fit_transform(example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()
