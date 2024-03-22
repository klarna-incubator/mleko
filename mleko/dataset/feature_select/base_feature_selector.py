"""Module for the base feature selector class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.cache.fingerprinters import DictFingerprinter, VaexFingerprinter
from mleko.cache.handlers import JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseFeatureSelector(LRUCacheMixin, ABC):
    """Abstract class for feature selection.

    The feature selection process is implemented in the `fit`, `transform`, and `fit_transform` methods, following the
    scikit-learn API. The `fit` method takes a DataFrame as input and returns a fitted feature selector. The `transform`
    method takes a DataFrame as input and returns a transformed DataFrame. The `fit_transform` method takes a DataFrame
    as input and returns a tuple of a fitted feature selector and a transformed DataFrame.

    Note:
        The default set of features to be used by the feature selector is all features applicable to the feature
        selector. This can be overridden by passing a list of feature names to the `features` parameter of the
        constructor. The default set of features to be ignored by the feature selector is no features. This can be
        overridden by passing a list of feature names to the `ignore_features` parameter of the constructor.
    """

    def __init__(
        self,
        features: list[str] | tuple[str, ...] | None,
        ignore_features: list[str] | tuple[str, ...] | None,
        cache_directory: str | Path,
        cache_size: int,
    ) -> None:
        """Initializes the feature selector and ensures the destination directory exists.

        Note:
            The `features` and `ignore_features` arguments are mutually exclusive. If both are specified, a
            `ValueError` is raised.

        Args:
            features: List of feature names to be used by the feature selector. If None, the default is all features
                applicable to the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector. If None, the default is to
                ignore no features.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of cache entries.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        super().__init__(cache_directory, cache_size)
        if features is not None and ignore_features is not None:
            msg = "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            logger.error(msg)
            raise ValueError(msg)

        self._features: tuple[str, ...] | None = tuple(features) if features is not None else None
        self._ignore_features: tuple[str, ...] = tuple(ignore_features) if ignore_features is not None else tuple()
        self._feature_selector = None

    def fit(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, Any]:
        """Fits the feature selector to the specified DataFrame, using the cached result if available.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame to be fitted.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting to be recomputed even if the result is cached.
            disable_cache: If set to True, disables the cache.

        Returns:
            Updated DataSchema and fitted feature selector.
        """
        ds, feature_selector = self._cached_execute(
            lambda_func=lambda: self._fit(data_schema, dataframe),
            cache_key_inputs=[
                self._fingerprint(),
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=JOBLIB_CACHE_HANDLER,
            disable_cache=disable_cache,
        )
        self._assign_feature_selector(feature_selector)
        return ds, feature_selector

    def transform(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, vaex.DataFrame]:
        """Extracts the selected features from the DataFrame, using the cached result if available.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame to be transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.
            disable_cache: If set to True, disables the cache.

        Raises:
            RuntimeError: If the feature selector has not been fitted.

        Returns:
            Updated DataSchema and transformed DataFrame.
        """
        if self._feature_selector is None:
            msg = "Feature selector must be fitted before it can be used to extract selected features."
            logger.error(msg)
            raise RuntimeError(msg)

        ds, df = self._cached_execute(
            lambda_func=lambda: self._transform(data_schema, dataframe),
            cache_key_inputs=[
                self._fingerprint(),
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER],
            disable_cache=disable_cache,
        )
        return ds, df

    def fit_transform(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, Any, vaex.DataFrame]:
        """Fits the feature selector to the specified DataFrame and extracts the selected features from the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame to be fitted and transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting and transformation to be recomputed even if the result is
                cached.
            disable_cache: If set to True, disables the cache.

        Returns:
            Tuple of updated DataSchema, fitted feature selector, and transformed DataFrame.
        """
        ds, feature_selector, df = self._cached_execute(
            lambda_func=lambda: self._fit_transform(data_schema, dataframe),
            cache_key_inputs=[
                self._fingerprint(),
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[JOBLIB_CACHE_HANDLER, JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER],
            disable_cache=disable_cache,
        )
        self._assign_feature_selector(feature_selector)
        return ds, feature_selector, df

    def _fit_transform(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, Any, vaex.DataFrame]:
        """Fits the feature selector to the specified DataFrame and extracts the selected features from the DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame used for feature selection.

        Returns:
            Tuple of updated DataSchema, fitted feature selector, and transformed DataFrame.
        """
        ds, feature_selector = self._fit(data_schema, dataframe)
        ds, df = self._transform(data_schema, dataframe)
        return ds, feature_selector, df

    def _assign_feature_selector(self, feature_selector: Any) -> None:
        """Assigns the specified feature selector to the feature_selector attribute.

        Can be overridden by subclasses to assign the feature selector using a different method.

        Args:
            feature_selector: Feature selector to be assigned.
        """
        self._feature_selector = feature_selector

    def _feature_set(self, data_schema: DataSchema) -> list[str]:
        """Returns the list of features to be used by the feature selector.

        It is the default set of features minus the features to be ignored if the `features` argument is None, or the
        list of names in the `features` argument if it is not None.

        Args:
            data_schema: DataSchema of the DataFrame.

        Returns:
            Sorted list of feature names to be used by the feature selector.
        """
        return sorted(
            set(self._default_features(data_schema)) - set(self._ignore_features)
            if self._features is None
            else self._features
        )

    @abstractmethod
    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, Any]:
        """Fits the feature selector to the specified DataFrame.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame to be fitted.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the dataframe by extracting the selected features.

        Args:
            data_schema: DataSchema of the DataFrame.
            dataframe: DataFrame to be transformed.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:
        """Returns the default set of features to be used by the feature selector.

        Args:
            data_schema: DataSchema of the DataFrame.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.
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
