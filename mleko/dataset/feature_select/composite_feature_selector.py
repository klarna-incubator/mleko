"""A feature selector that combines multiple feature selectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_feature_selector import BaseFeatureSelector


logger = CustomLogger()
"""A module-level logger for the module."""


class CompositeFeatureSelector(BaseFeatureSelector):
    """A feature selector that combines multiple feature selectors.

    It is possible to combine multiple feature selectors into a single feature selector. This can be useful when
    multiple feature selectors need to be applied to a DataFrame and the cache needs to be shared between them.
    """

    @auto_repr
    def __init__(
        self,
        feature_selectors: list[BaseFeatureSelector] | tuple[BaseFeatureSelector, ...],
        cache_directory: str | Path = "data/composite-feature-selector",
        cache_size: int = 1,
    ) -> None:
        """Initializes the composite feature selector.

        The composite feature selector will combine the feature selectors into a single feature selector. Each feature
        selector will be applied to the DataFrame in the order they are specified.

        Args:
            feature_selectors: List of feature selectors to be combined.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import CompositeFeatureSelector, MissingRateFeatureSelector
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 2, 3, 4, None, None, None, None, None, None],
            ...     c=[1, 2, 3, 4, 5, 6, None, None, None, None],
            ... )
            >>> ds = DataSchema(
            ...     numerical=["a", "b", "c"],
            ... )
            >>> feature_selector = CompositeFeatureSelector(
            ...     feature_selectors=[
            ...         MissingRateFeatureSelector(
            ...             missing_rate_threshold=0.75,
            ...         ),
            ...         MissingRateFeatureSelector(
            ...             missing_rate_threshold=0.50,
            ...         ),
            ...     ],
            ... )
            >>> ds, _, df = feature_selector.fit_transform(ds, df)
            >>> df
            #    a    c
            0    1    1
            1    2    2
            2    3    3
            3    4    4
            4    5    5
            5    6    6
            6    7    None
            7    8    None
            8    9    None
            9   10    None
        """
        super().__init__(None, None, cache_directory, cache_size)
        self._feature_selectors = tuple(feature_selectors)
        self._feature_selector: list[Any] = []

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, list[Any]]:
        """Fits the feature selector on the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame on which the feature selector will be fitted.

        Returns:
            Tuple of updated DataSchema and list of fitted feature selectors.
        """
        feature_selectors: list[Any] = []
        for i, feature_selector in enumerate(self._feature_selectors):
            logger.info(
                f"Fitting composite feature selection step {i+1}/{len(self._feature_selectors)}: "
                f"{feature_selector.__class__.__name__}."
            )
            data_schema, feature_selector = feature_selector.fit(data_schema, dataframe, disable_cache=True)
            feature_selectors.append(feature_selector)
            logger.info(f"Finished fitting composite feature selection step {i+1}/{len(self._feature_selectors)}.")
        return data_schema, feature_selectors

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Selects the features from the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame from which the features will be selected.

        Returns:
            DataFrame with the selected features.
        """
        for i, feature_selector in enumerate(self._feature_selectors):
            logger.info(
                f"Executing composite feature selection step {i+1}/{len(self._feature_selectors)}: "
                f"{feature_selector.__class__.__name__}."
            )
            data_schema, dataframe = feature_selector.transform(data_schema, dataframe, disable_cache=True)
            dataframe = dataframe.extract()
            logger.info(f"Finished composite feature selection step {i+1}/{len(self._feature_selectors)}.")
        return data_schema, dataframe

    def _fit_transform(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, Any, vaex.DataFrame]:
        """Fits the feature selector to the specified DataFrame and extracts the selected features from the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame used for feature selection.

        Returns:
            Tuple of updated data schema, fitted feature selector and transformed DataFrame.
        """
        feature_selectors: list[Any] = []
        for i, feature_selector in enumerate(self._feature_selectors):
            logger.info(
                f"Executing composite feature selection step {i+1}/{len(self._feature_selectors)}: "
                f"{feature_selector.__class__.__name__}."
            )
            data_schema, feature_selector, dataframe = feature_selector.fit_transform(
                data_schema, dataframe, disable_cache=True
            )
            feature_selectors.append(feature_selector)
            dataframe = dataframe.extract()
            logger.info(
                "Finished fitting and transforming composite feature "
                f"selection step {i+1}/{len(self._feature_selectors)}."
            )
        return data_schema, feature_selectors, dataframe

    def _assign_feature_selector(self, feature_selector: Any) -> None:
        """Assigns the specified feature selector to the feature_selector attribute.

        Can be overridden by subclasses to assign the feature selector using a different method.

        Args:
            feature_selector: Feature selector to be assigned.
        """
        self._feature_selector = feature_selector
        for feature_selector_obj, fitted_feature_selector in zip(self._feature_selectors, feature_selector):
            feature_selector_obj._feature_selector = fitted_feature_selector

    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:  # pragma: no cover
        """Returns the default features of the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.

        Returns:
            Tuple of default features.
        """
        features = data_schema.get_features()
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the feature selector.

        Returns:
            Fingerprint of the feature selector.
        """
        return super()._fingerprint(), tuple(
            feature_selector._fingerprint() for feature_selector in self._feature_selectors
        )
