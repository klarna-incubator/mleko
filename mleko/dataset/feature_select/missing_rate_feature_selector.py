"""Module for the missing rate feature selector."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
from tqdm.auto import tqdm

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_columns

from .base_feature_selector import BaseFeatureSelector


logger = CustomLogger()
"""A module-level logger for the module."""


class MissingRateFeatureSelector(BaseFeatureSelector):
    """Selects features based on the missing rate."""

    @auto_repr
    def __init__(
        self,
        missing_rate_threshold: float,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        cache_directory: str | Path = "data/missing-rate-feature-selector",
        cache_size: int = 1,
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
            missing_rate_threshold: The maximum missing rate allowed for a feature to be selected.
            features: List of feature names to be used by the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import MissingRateFeatureSelector
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 2, 3, 4, 5, None, None, None, None, None],
            ...     c=[1, 2, 3, 4, 5, 6, None, None, None, None],
            ... )
            >>> ds = DataSchema(numerical=["a", "b", "c"])
            >>> ds, _, df = MissingRateFeatureSelector(
            ...     ignore_features=["c"],
            ...     missing_rate_threshold=0.3,
            ... ).fit_transform(ds, df)
            >>> df.get_column_names()
            ['a', 'b']
        """
        super().__init__(features, ignore_features, cache_directory, cache_size)
        self._missing_rate_threshold = missing_rate_threshold
        self._feature_selector: set[str] = set()

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, set[str]]:
        """Fits the feature selector on the input data.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to fit the feature selector on.

        Returns:
            Updated DataSchema and the set of features with a missing rate above the threshold.
        """
        features = self._feature_set(data_schema)
        logger.info(f"Fitting missing rate feature selector on {len(features)} features: {features}.")

        missing_rate: dict[str, float] = {}
        for feature in tqdm(features, desc="Calculating missing rates for features"):
            column = get_column(dataframe, feature)
            missing_rate[feature] = column.countna() / dataframe.shape[0]

        self._feature_selector = {
            feature for feature in features if missing_rate[feature] >= self._missing_rate_threshold
        }
        ds = data_schema.copy().drop_features(self._feature_selector)

        return ds, self._feature_selector

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Selects features based on the missing rate.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to select features from.

        Returns:
            The DataFrame with the selected features.
        """
        dropped_features = self._feature_selector
        logger.info(
            f"Dropping ({len(dropped_features)}) features with missing rate >= {self._missing_rate_threshold}: "
            f"{dropped_features}."
        )
        selected_features = [feature for feature in dataframe.get_column_names() if feature not in dropped_features]
        ds = data_schema.copy().drop_features(dropped_features)

        return ds, get_columns(dataframe, selected_features)

    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:
        """Returns the default set of features.

        Args:
            data_schema: The DataSchema of the DataFrame.

        Returns:
            Tuple of default features.
        """
        features = data_schema.get_features()
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the feature selector.

        Appends the missing rate threshold to the fingerprint.

        Returns:
            The fingerprint of the feature selector.
        """
        return super()._fingerprint(), self._missing_rate_threshold
