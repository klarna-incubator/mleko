"""Module for the variance feature selector."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
from tqdm.auto import tqdm
from vaex.ml import MaxAbsScaler

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_columns


logger = CustomLogger()
"""A module-level logger for the module."""


class VarianceFeatureSelector(BaseFeatureSelector):
    """Selects features based on the variance."""

    @auto_repr
    def __init__(
        self,
        variance_threshold: float,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        cache_directory: str | Path = "data/variance-feature-selector",
        cache_size: int = 1,
    ) -> None:
        """Initializes the feature selector.

        The feature selector will select all features with a variance above the specified threshold.
        The default set of features is all numeric features in the DataFrame.

        Note:
            Only works with numeric features.

        Warning:
            Make sure to ignore any important features that need to be kept, such as the
            target feature or some identifier.

        Args:
            variance_threshold: The minimum variance allowed for a feature to be selected.
            features: List of feature names to be used by the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import VarianceFeatureSelector
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     c=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            ...     d=["str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9", "str10"],
            ... )
            >>> ds = DataSchema(
            ...     numerical=["a", "b", "c"],
            ...     categorical=["d"],
            ... )
            >>> selector = VarianceFeatureSelector(
            ...     variance_threshold=0.1,
            ...     ignore_features=["c"],
            ... )
            >>> ds, _, df = selector.fit_transform(ds, df)
            >>> df.get_column_names()
            ['a', 'c', 'd']
        """
        super().__init__(features, ignore_features, cache_directory, cache_size)
        self._variance_threshold = variance_threshold
        self._feature_selector: set[str] = set()

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, set[str]]:
        """Fits the feature selector on the input data.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to fit the feature selector on.

        Returns:
            Updated DataSchema and the set of features with a variance below the threshold.
        """
        features = self._feature_set(data_schema)
        logger.info(f"Fitting variance feature selector on {len(features)} features: {features}.")

        if self._variance_threshold > 0:
            scaler = MaxAbsScaler(features=list(features), prefix="")
            df = scaler.fit_transform(dataframe)
        else:
            df = dataframe

        variance: dict[str, float] = {}
        for feature in tqdm(features, desc="Calculating variance for features"):
            column = get_column(df, feature)
            variance[feature] = column.var()

        self._feature_selector = {feature for feature in features if variance[feature] <= self._variance_threshold}
        ds = data_schema.copy().drop_features(self._feature_selector)

        return ds, self._feature_selector

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Selects features based on the variance.

        Args:
            data_schema: The DataSchema of the DataFrame.
            dataframe: The DataFrame to select features from.

        Returns:
            Updated DataSchema and DataFrame with the selected features.
        """
        dropped_features = self._feature_selector
        logger.info(
            f"Dropping ({len(dropped_features)}) features with normalized variance <= {self._variance_threshold}: "
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
        features = data_schema.get_features(["numerical"])
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns a hashable fingerprint of the feature selector.

        Append the variance threshold to the fingerprint.

        Returns:
            The fingerprint of the feature selector.
        """
        return super()._fingerprint(), self._variance_threshold
