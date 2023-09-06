"""Module for the base transformer class."""
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


class BaseTransformer(LRUCacheMixin, ABC):
    """Abstract class for feature transformation.

    The feature transformation process is implemented in the `fit`, `transform`, and `fit_transform` methods. The
    `fit` method fits the transformer to the specified DataFrame, the `transform` method transforms the specified
    features in the DataFrame, and the `fit_transform` method fits the transformer to the specified DataFrame and
    transforms the specified features in the DataFrame.
    """

    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        cache_size: int,
    ) -> None:
        """Initializes the transformer and ensures the destination directory exists.

        Args:
            cache_directory: Directory where the cache will be stored locally.
            features: List of feature names to be used by the transformer.
            cache_size: The maximum number of cache entries to keep in the cache.
        """
        super().__init__(cache_directory, cache_size)
        self._features: tuple[str, ...] = tuple(features)
        self._transformer = None

    def fit(self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False) -> Any:
        """Fits the transformer to the specified DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame to be fitted.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting to be recomputed even if the result is cached.

        Returns:
            Fitted transformer.
        """
        transformer = self._cached_execute(
            lambda_func=lambda: self._fit(dataframe),
            cache_key_inputs=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=JOBLIB_CACHE_HANDLER,
        )
        self._assign_transformer(transformer)
        return transformer

    def transform(
        self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False
    ) -> vaex.DataFrame:
        """Transforms the specified features in the DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame to be transformed.
            cache_group: The cache group to use.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.

        Raises:
            RuntimeError: If the transformer has not been fitted.

        Returns:
            Transformed DataFrame.
        """
        if self._transformer is None:
            raise RuntimeError("Transformer must be fitted before it can be used to transform data.")

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
        """Fits the transformer to the specified DataFrame and transforms the specified features in the DataFrame.

        Args:
            dataframe: DataFrame used for fitting and transformation.
            cache_group: The cache group to use.
            force_recompute: Whether to force the fitting and transformation to be recomputed even if the result is

        Returns:
            Transformed DataFrame.
        """
        transformer, df = self._cached_execute(
            lambda_func=lambda: self._fit_transform(dataframe),
            cache_key_inputs=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER],
        )
        self._assign_transformer(transformer)
        return transformer, df

    def _fit_transform(self, dataframe: vaex.DataFrame) -> tuple[Any, vaex.DataFrame]:
        """Fits the transformer to the specified DataFrame and transforms the specified features in the DataFrame.

        Args:
            dataframe: DataFrame used for fitting and transformation.

        Returns:
            Fitted transformer and transformed DataFrame.
        """
        transformer = self._fit(dataframe)
        return transformer, self._transform(dataframe)

    def _assign_transformer(self, transformer: Any) -> None:
        """Assigns the specified transformer to the transformer attribute.

        Can be overridden by subclasses to assign the transformer using a different method.

        Args:
            transformer: Transformer to be assigned.
        """
        self._transformer = transformer

    @abstractmethod
    def _fit(self, dataframe: vaex.DataFrame) -> Any:
        """Fits the transformer to the specified DataFrame.

        Args:
            dataframe: DataFrame to be fitted.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the specified features in the DataFrame.

        Args:
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
