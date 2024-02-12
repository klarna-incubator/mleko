"""Module for the Pearson correlation feature selector.""" ""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Hashable

import numpy as np
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_columns

from .base_feature_selector import BaseFeatureSelector


logger = CustomLogger()
"""A module-level logger for the module."""


class PearsonCorrelationFeatureSelector(BaseFeatureSelector):
    """Selects features based on the Pearson correlation."""

    @auto_repr
    def __init__(
        self,
        correlation_threshold: float,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        cache_directory: str | Path = "data/pearson-correlation-feature-selector",
        cache_size: int = 1,
    ) -> None:
        """Initializes the feature selector.

        Will drop one of two features that are highly correlated. The feature to be dropped is the one with the lowest
        average correlation with all other features. If both features have the same average correlation, the first
        feature will be dropped. The default set of features is all numeric features in the DataFrame.

        Note:
            Only works with numeric features.

        Warning:
            Make sure to ignore any important features that need to be kept, such as the
            target feature or some identifier.

        Args:
            correlation_threshold: The maximum correlation allowed for a feature to be selected.
            features: List of feature names to be used by the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import PearsonCorrelationFeatureSelector
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            ...     c=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ... )
            >>> ds = DataSchema(numerical=["a", "b", "c"])
            >>> feature_selector = PearsonCorrelationFeatureSelector(
            ...     correlation_threshold=0.75,
            ... )
            >>> ds, _, df = feature_selector.fit_transform(ds, df)
            >>> df.get_column_names()
            ['a', 'c']
        """
        super().__init__(features, ignore_features, cache_directory, cache_size)
        self._correlation_threshold = correlation_threshold
        self._feature_selector: set[str] = set()

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, set[str]]:
        """Fits the feature selector on the input data.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to fit the feature selector on.

        Returns:
            Updated DataSchema and the set of features to be dropped.
        """
        features = self._feature_set(data_schema)
        logger.info(f"Fitting pearson correlation feature selector on {len(features)} features: {features}.")

        corr_matrix = abs(np.array(dataframe.correlation(features)))
        avg_corr = corr_matrix.mean(axis=1)

        # Generate all possible pairs of features
        feature_pairs = list(itertools.combinations(enumerate(features), 2))

        # Find correlated feature pairs and the feature to drop based on average correlation
        correlated_features = [
            (f_a, f_b, f_a if avg_corr[idx_a] > avg_corr[idx_b] else f_b)
            for (idx_a, f_a), (idx_b, f_b) in feature_pairs
            if corr_matrix[idx_a, idx_b] >= self._correlation_threshold
        ]

        # Create sets of all correlated features and potentially dropped features
        all_correlated_features = {f_a for f_a, _, _ in correlated_features}.union(
            f_b for _, f_b, _ in correlated_features
        )
        potentially_dropped = {drop_f for _, _, drop_f in correlated_features}

        # Identify features guaranteed to be kept
        guaranteed_kept = all_correlated_features.difference(potentially_dropped)

        # Identify features guaranteed to be dropped
        guaranteed_dropped = {
            f
            for f_a, f_b, _ in correlated_features
            if f_a in guaranteed_kept or f_b in guaranteed_kept
            for f in (f_a, f_b)
            if f not in guaranteed_kept
        }

        # Identify features that are not guaranteed to be dropped but are highly correlated
        possible_drop = potentially_dropped.difference(guaranteed_dropped)

        # Identify additional features to be dropped based on correlation with possible_drop features
        additional_dropped = {
            drop_f
            for f_a, f_b, drop_f in correlated_features
            if (f_a in possible_drop or f_b in possible_drop)
            and (f_a not in guaranteed_dropped and f_b not in guaranteed_dropped)
        }

        # Combine guaranteed_dropped and additional_dropped sets
        self._feature_selector = guaranteed_dropped.union(additional_dropped)
        ds = data_schema.copy().drop_features(self._feature_selector)

        return ds, self._feature_selector

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Selects features based on the Pearson correlation.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to select features from.

        Returns:
            The updated DataSchema and DataFrame with the selected features.
        """
        dropped_features = self._feature_selector
        logger.info(
            f"Dropping ({len(dropped_features)}) features with correlation >= {self._correlation_threshold}: "
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
        features = data_schema.get_features(["numerical", "boolean"])
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns a hashable fingerprint of the feature selector.

        Append the pearson correlation threshold to the fingerprint.

        Returns:
            The fingerprint of the feature selector.
        """
        return super()._fingerprint(), self._correlation_threshold
