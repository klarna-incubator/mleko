"""Module for the missing rate feature selector."""
from __future__ import annotations

from pathlib import Path

import vaex
from tqdm import tqdm

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.cache.format.vaex_arrow_cache_format_mixin import VaexArrowCacheFormatMixin
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.vaex_helpers import get_column, get_columns

from .base_feature_selector import BaseFeatureSelector


logger = CustomLogger()
"""The logger for the module."""


class MissingRateFeatureSelector(BaseFeatureSelector, VaexArrowCacheFormatMixin, LRUCacheMixin):
    """Selects features based on the missing rate."""

    def __init__(
        self,
        output_directory: str | Path,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        missing_rate_threshold: float = 0.7,
        max_cache_entries: int = 1,
    ) -> None:
        """Initializes the feature selector.

        The feature selector will select all features with a missing rate below the specified threshold. The default
        set of features is all features in the DataFrame.

        Note:
            Works with all types of features.

        Warning:
            Make sure to ignore any important features that need to be kept, such as the
            target feature or some identifier.

        Args:
            output_directory: Directory where the selected features will be stored locally.
            features: List of feature names to be used by the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector.
            missing_rate_threshold: The maximum missing rate allowed for a feature to be selected.
            max_cache_entries: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import MissingRateFeatureSelector
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 2, 3, 4, 5, None, None, None, None, None],
            ...     c=[1, 2, 3, 4, 5, 6, None, None, None, None],
            ... )
            >>> MissingRateFeatureSelector(
            ...     output_directory=".",
            ...     features=["a", "b", "e"],
            ...     ignore_features=["c"],
            ...     missing_rate_threshold=0.3,
            ... ).select_features(df).get_column_names()
            ['a', 'b']
        """
        BaseFeatureSelector.__init__(self, output_directory, features, ignore_features)
        VaexArrowCacheFormatMixin.__init__(self)
        LRUCacheMixin.__init__(self, output_directory, VaexArrowCacheFormatMixin.cache_file_suffix, max_cache_entries)
        self._missing_rate_threshold = missing_rate_threshold

    def select_features(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Selects features based on the missing rate.

        Will cache the result of the feature selection.

        Args:
            dataframe: The DataFrame to select features from.
            force_recompute: Whether to force recompute the feature selection.

        Returns:
            The DataFrame with the selected features.
        """
        return self._cached_execute(
            lambda_func=lambda: self._select_features(dataframe),
            cache_keys=[
                self._features,
                self._ignore_features,
                self._missing_rate_threshold,
                (dataframe, VaexFingerprinter()),
            ],
            force_recompute=force_recompute,
        )

    def _select_features(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Selects features based on the missing rate.

        Args:
            dataframe: The DataFrame to select features from.

        Returns:
            The DataFrame with the selected features.
        """
        feature_names = self._feature_set(dataframe)
        logger.info(f"Selecting features from the following set: {feature_names}")

        missing_rate: dict[str, float] = {}
        for feature_name in tqdm(feature_names, desc="Calculating missing rates for features"):
            column = get_column(dataframe, feature_name)
            missing_rate[feature_name] = column.countna() / dataframe.shape[0]

        dropped_features = {
            feature_name for feature_name in feature_names if missing_rate[feature_name] >= self._missing_rate_threshold
        }
        logger.info(f"Dropping features with missing rate >= {self._missing_rate_threshold}: {dropped_features}")

        selected_features = [
            feature_name for feature_name in dataframe.get_column_names() if feature_name not in dropped_features
        ]
        return get_columns(dataframe, selected_features)

    def _default_features(self, dataframe: vaex.DataFrame) -> frozenset[str]:
        """Returns the default set of features.

        Args:
            dataframe: The DataFrame to select features from.

        Returns:
            The default set of features.
        """
        feature_names = dataframe.get_column_names()
        return frozenset(str(feature_name) for feature_name in feature_names)
