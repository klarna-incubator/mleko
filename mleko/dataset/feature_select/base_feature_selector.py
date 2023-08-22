"""Module for the base feature selector class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.cache.handlers.joblib_cache_handler import JOBLIB_CACHE_HANDLER
from mleko.cache.handlers.vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseFeatureSelector(LRUCacheMixin, ABC):
    """Abstract class for feature selection.

    The feature selection process is implemented in the `select_features` method, which takes a DataFrame as input and
    returns a list of paths to the selected features.

    Note:
        The default set of features to be used by the feature selector is all features applicable to the feature
        selector. This can be overridden by passing a list of feature names to the `features` parameter of the
        constructor. The default set of features to be ignored by the feature selector is no features. This can be
        overridden by passing a list of feature names to the `ignore_features` parameter of the constructor.
    """

    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...] | None,
        ignore_features: list[str] | tuple[str, ...] | None,
        cache_size: int,
    ) -> None:
        """Initializes the feature selector and ensures the destination directory exists.

        Note:
            The `features` and `ignore_features` arguments are mutually exclusive. If both are specified, a
            `ValueError` is raised.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the feature selector. If None, the default is all features
                applicable to the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector. If None, the default is to
                ignore no features.
            cache_size: The maximum number of cache entries.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        LRUCacheMixin.__init__(self, cache_directory, cache_size)
        if features is not None and ignore_features is not None:
            error_msg = (
                "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._features: tuple[str, ...] | None = tuple(features) if features is not None else None
        self._ignore_features: tuple[str, ...] = tuple(ignore_features) if ignore_features is not None else tuple()
        self._feature_selector = None

    def fit(self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False) -> Any:
        """Fits the feature selector to the specified DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame to be fitted.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting to be recomputed even if the result is cached.

        Returns:
            Fitted feature selector.
        """
        feature_selector = self._cached_execute(
            lambda_func=lambda: self._fit(dataframe),
            cache_key_inputs=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=JOBLIB_CACHE_HANDLER,
        )
        self._assign_feature_selector(feature_selector)
        return feature_selector

    def transform(
        self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False
    ) -> vaex.DataFrame:
        """Extracts the selected features from the DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame to be transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.

        Raises:
            RuntimeError: If the feature selector has not been fitted.

        Returns:
            Transformed DataFrame.
        """
        if self._feature_selector is None:
            raise RuntimeError("Feature selector must be fitted before it can be used to extract selected features.")

        return self._cached_execute(
            lambda_func=lambda: self._transform(dataframe),
            cache_key_inputs=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=VAEX_DATAFRAME_CACHE_HANDLER,
        )

    def fit_transform(
        self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False
    ) -> tuple[Any, vaex.DataFrame]:
        """Fits the feature selector to the specified DataFrame and extracts the selected features from the DataFrame.

        Args:
            dataframe: DataFrame to be fitted and transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting and transformation to be recomputed even if the result is
                cached.

        Returns:
            Tuple of fitted feature selector and transformed DataFrame.
        """
        feature_selector, df = self._cached_execute(
            lambda_func=lambda: self._fit_transform(dataframe),
            cache_key_inputs=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER],
        )
        self._assign_feature_selector(feature_selector)
        return feature_selector, df

    def _fit_transform(self, dataframe: vaex.DataFrame) -> tuple[Any, vaex.DataFrame]:
        """Fits the feature selector to the specified DataFrame and extracts the selected features from the DataFrame.

        Args:
            dataframe: DataFrame used for feature selection.

        Returns:
            Tuple of fitted feature selector and transformed DataFrame.
        """
        fitted_feature_selector = self._fit(dataframe)
        return fitted_feature_selector, self._transform(dataframe)

    def _assign_feature_selector(self, feature_selector: Any) -> None:
        """Assigns the specified feature selector to the feature_selector attribute.

        Can be overridden by subclasses to assign the feature selector using a different method.

        Args:
            feature_selector: Feature selector to be assigned.
        """
        self._feature_selector = feature_selector

    def _feature_set(self, dataframe: vaex.DataFrame) -> list[str]:
        """Returns the list of features to be used by the feature selector.

        It is the default set of features minus the features to be ignored if the `features` argument is None, or the
        list of names in the `features` argument if it is not None.

        Args:
            dataframe: DataFrame from which to select features.

        Returns:
            Sorted list of feature names to be used by the feature selector.
        """
        return sorted(
            set(self._default_features(dataframe)) - set(self._ignore_features)
            if self._features is None
            else self._features
        )

    @abstractmethod
    def _fit(self, dataframe: vaex.DataFrame) -> Any:
        """Fits the feature selector to the specified DataFrame.

        Args:
            dataframe: DataFrame to be fitted.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            Fitted feature selector.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the dataframe by extracting the selected features.

        Args:
            dataframe: DataFrame to be transformed.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            Transformed DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_features(self, dataframe: vaex.DataFrame) -> tuple[str, ...]:
        """Returns the default set of features to be used by the feature selector.

        Args:
            dataframe: DataFrame from which to select features.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            Tuple of feature names to be used by the feature selector.
        """
        raise NotImplementedError

    @abstractmethod
    def _fingerprint(self) -> Hashable:
        """Returns a hashable object that uniquely identifies the feature selector.

        Note:
            Subclasses should call the parent method and include the result in the hashable object along with any other
            information that uniquely identifies the feature selector. All attributes that are used in the
            feature selector that affect the output of the feature selector should be included in the hashable object.

        Returns:
            Hashable object that uniquely identifies the feature selector.
        """
        return self.__class__.__name__, self._features, self._ignore_features
