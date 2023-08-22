"""A feature selector that combines multiple feature selectors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable

import vaex

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
        cache_directory: str | Path,
        feature_selectors: list[BaseFeatureSelector] | tuple[BaseFeatureSelector, ...],
        cache_size: int = 1,
    ) -> None:
        """Initializes the composite feature selector.

        The composite feature selector will combine the feature selectors into a single feature selector. Each feature
        selector will be applied to the DataFrame in the order they are specified.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            feature_selectors: List of feature selectors to be combined.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.feature_select import CompositeFeatureSelector, MissingRateFeatureSelector
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[1, 2, 3, 4, None, None, None, None, None, None],
            ...     c=[1, 2, 3, 4, 5, 6, None, None, None, None],
            ... )
            >>> feature_selector = CompositeFeatureSelector(
            ...     cache_directory=".",
            ...     feature_selectors=[
            ...         MissingRateFeatureSelector(
            ...             cache_directory=".",
            ...             missing_rate_threshold=0.75,
            ...         ),
            ...         MissingRateFeatureSelector(
            ...             cache_directory=".",
            ...             missing_rate_threshold=0.50,
            ...         ),
            ...     ],
            ... )
            >>> _, df = feature_selector.fit_transform(df)
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
        super().__init__(cache_directory, None, None, cache_size)
        self._feature_selectors = tuple(feature_selectors)
        self._feature_selector: list[Any] = []

    def _fit(self, dataframe: vaex.DataFrame) -> list[Any]:
        """Fits the feature selector on the DataFrame.

        Args:
            dataframe: DataFrame on which the feature selector will be fitted.

        Returns:
            List of fitted feature selectors.
        """
        feature_selectors: list[Any] = []
        for i, feature_selector in enumerate(self._feature_selectors):
            logger.info(
                f"Fitting composite feature selection step {i+1}/{len(self._feature_selectors)}: "
                f"{feature_selector.__class__.__name__}."
            )
            feature_selector = feature_selector._fit(dataframe)
            feature_selectors.append(feature_selector)
            logger.info(f"Finished fitting composite feature selection step {i+1}/{len(self._feature_selectors)}.")
        return feature_selectors

    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Selects the features from the DataFrame.

        Args:
            dataframe: DataFrame from which the features will be selected.

        Returns:
            DataFrame with the selected features.
        """
        for i, feature_selector in enumerate(self._feature_selectors):
            logger.info(
                f"Executing composite feature selection step {i+1}/{len(self._feature_selectors)}: "
                f"{feature_selector.__class__.__name__}."
            )
            dataframe = feature_selector._transform(dataframe).extract()
            logger.info(f"Finished composite feature selection step {i+1}/{len(self._feature_selectors)}.")
        return dataframe

    def _assign_feature_selector(self, feature_selector: Any) -> None:
        """Assigns the specified feature selector to the feature_selector attribute.

        Can be overridden by subclasses to assign the feature selector using a different method.

        Args:
            feature_selector: Feature selector to be assigned.
        """
        self._feature_selector = feature_selector
        for feature_selector_obj, fitted_feature_selector in zip(self._feature_selectors, feature_selector):
            feature_selector_obj._feature_selector = fitted_feature_selector

    def _default_features(self, dataframe: vaex.DataFrame) -> tuple[str, ...]:  # pragma: no cover
        """Returns the default features of the DataFrame.

        Args:
            dataframe: DataFrame from which the default features will be extracted.

        Returns:
            Tuple of default features.
        """
        features = dataframe.get_column_names()
        return tuple(str(feature) for feature in features)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the feature selector.

        Returns:
            Fingerprint of the feature selector.
        """
        return super()._fingerprint(), tuple(
            feature_selector._fingerprint() for feature_selector in self._feature_selectors
        )
