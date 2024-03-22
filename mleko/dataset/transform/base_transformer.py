"""Module for the base transformer class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.cache.fingerprinters import DictFingerprinter, VaexFingerprinter
from mleko.cache.handlers.joblib_cache_handler import JOBLIB_CACHE_HANDLER
from mleko.cache.handlers.vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseTransformer(LRUCacheMixin, ABC):
    """Abstract class for feature transformation.

    The feature transformation process is implemented in the `fit`, `transform`, and `fit_transform` methods. The
    `fit` method fits the transformer to the specified DataFrame, the `transform` method transforms the specified
    features in the DataFrame, and the `fit_transform` method fits the transformer to the specified DataFrame and
    transforms the specified features in the DataFrame.

    Warning:
        The _transformer attribute is not set by the base class. Subclasses must place all transformer-related logic
        inside the attribute to correctly handle caching and ensure that the transformer is correctly assigned. For
        example, the `fit` method should assign the fitted transformer to the _transformer attribute, and the
        `transform` method should use the _transformer attribute to transform the DataFrame.
    """

    def __init__(
        self,
        features: list[str] | tuple[str, ...],
        cache_directory: str | Path,
        cache_size: int,
    ) -> None:
        """Initializes the transformer and ensures the destination directory exists.

        Args:
            features: List of feature names to be used by the transformer.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of cache entries to keep in the cache.
        """
        super().__init__(cache_directory, cache_size)
        self._features: tuple[str, ...] = tuple(features)
        self._transformer = None

    def fit(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, Any]:
        """Fits the transformer to the specified DataFrame, using the cached result if available.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be fitted.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting to be recomputed even if the result is cached.
            disable_cache: If set to True, disables the cache.

        Returns:
            Updated data schema and fitted transformer.
        """
        ds, transformer = self._cached_execute(
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
        self._assign_transformer(transformer)
        return ds, transformer

    def transform(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the specified features in the DataFrame, using the cached result if available.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.
            disable_cache: If set to True, disables the cache.

        Raises:
            RuntimeError: If the transformer has not been fitted.

        Returns:
            Updated data schema and transformed DataFrame.
        """
        if self._transformer is None:
            msg = "Transformer must be fitted before it can be used to transform data."
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
        """Fits the transformer to the specified DataFrame and transforms the specified features in the DataFrame.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame used for fitting and transformation.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting and transformation to be recomputed even if the result is
                cached.
            disable_cache: If set to True, disables the cache.

        Returns:
            Tuple of updated data schema, fitted transformer, and transformed DataFrame.
        """
        ds, transformer, df = self._cached_execute(
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
        self._assign_transformer(transformer)
        return ds, transformer, df

    def _fit_transform(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, Any, vaex.DataFrame]:
        """Fits the transformer to the specified DataFrame and transforms the specified features in the DataFrame.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame used for fitting and transformation.

        Returns:
            Tuple of updated data schema, fitted transformer, and transformed DataFrame.
        """
        ds, transformer = self._fit(data_schema, dataframe)
        ds, df = self._transform(data_schema, dataframe)
        return ds, transformer, df

    def _assign_transformer(self, transformer: Any) -> None:
        """Assigns the specified transformer to the transformer attribute.

        Can be overridden by subclasses to assign the transformer using a different method.

        Args:
            transformer: Transformer to be assigned.
        """
        self._transformer = transformer

    @abstractmethod
    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, Any]:
        """Fits the transformer to the specified DataFrame.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be fitted.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the specified features in the DataFrame.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be transformed.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _fingerprint(self) -> Hashable:
        """Returns a hashable object that uniquely identifies the transformer.

        The base implementation fingerprints the class name and the features used by the transformer.

        Note:
            Subclasses should call the parent method and include the result in the hashable object along with any
            other parameters that uniquely identify the transformer. All attributes that are used in the
            transformer that affect the result of the transformation should be included in the hashable object.

        Returns:
            Hashable object that uniquely identifies the transformer.
        """
        return self.__class__.__name__, self._features
