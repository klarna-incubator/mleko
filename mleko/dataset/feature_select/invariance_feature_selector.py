"""Module for the invariance feature selector."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
from tqdm.auto import tqdm

from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_columns


logger = CustomLogger()
"""A module-level logger for the module."""


class InvarianceFeatureSelector(BaseFeatureSelector):
    """Selects features based on invariance."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        cache_size: int = 1,
        disable_cache: bool = False,
    ) -> None:
        """Initializes the feature selector.

        The feature selector will filter out all invariant features. The default set of features
        are all categorical and boolean features in the DataFrame.

        Note:
            Only works with categorical and boolean features.

        Warning:
            Make sure to ignore any important features that need to be kept, such as the
            target feature or some identifier.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector.
            cache_size: The maximum number of entries to keep in the cache.
            disable_cache: Whether to disable caching.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import InvarianceFeatureSelector
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     c=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            ...     d=["str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9", "str10"],
            ... )
            >>> selector = InvarianceFeatureSelector(
            ...     cache_directory=".",
            ... )
            >>> df_selected = selector.select_features(df)
            >>> df_selected.get_column_names()
            ['a', 'c', 'd']
        """
        super().__init__(cache_directory, features, ignore_features, cache_size, disable_cache)
        self._feature_selector: set[str] = set()

    def _select_features(self, dataframe: vaex.DataFrame, fit: bool) -> vaex.DataFrame:
        """Selects features based on invariance.

        Args:
            dataframe: The DataFrame to select features from.
            fit: Whether to fit the feature selector on the input data.

        Returns:
            The DataFrame with the selected features.
        """
        if fit:
            self._fit(dataframe)

        dropped_features = self._feature_selector
        logger.info(f"Dropping ({len(dropped_features)}) invariant features: {dropped_features}.")
        selected_features = [feature for feature in dataframe.get_column_names() if feature not in dropped_features]
        return get_columns(dataframe, selected_features)

    def _fit(self, dataframe: vaex.DataFrame) -> None:
        """Fits the feature selector on the input data.

        Args:
            dataframe: The DataFrame to fit the feature selector on.
        """
        features = self._feature_set(dataframe)
        logger.info(f"Fitting invariance feature selector on {len(features)} features: {features}.")

        cardinality = {}
        for feature in tqdm(features, desc="Calculating invariance of features"):
            column = get_column(dataframe, feature)
            cardinality[feature] = column.nunique(limit=2, limit_raise=False)

        self._feature_selector = {feature for feature in features if cardinality[feature] == 1}

    def _default_features(self, dataframe: vaex.DataFrame) -> tuple[str, ...]:
        """Returns the default set of features.

        Args:
            dataframe: The DataFrame to select features from.

        Returns:
            Tuple of default features.
        """
        features = dataframe.get_column_names(dtype=["bool", "string"])
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the feature selector.

        Appends the missing rate threshold to the fingerprint.

        Returns:
            The fingerprint of the feature selector.
        """
        return super()._fingerprint()
