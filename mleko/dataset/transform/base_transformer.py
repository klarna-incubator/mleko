"""Module for the base transformer class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Hashable

import vaex

from mleko.cache.format.vaex_cache_format_mixin import VaexCacheFormatMixin
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseTransformer(VaexCacheFormatMixin, LRUCacheMixin, ABC):
    """Abstract class for feature transformation.

    The feature transformation process is implemented in the `transform` method, which takes a DataFrame as input and
    returns a transformed DataFrame.
    """

    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        cache_size: int,
    ) -> None:
        """Initializes the transformer and ensures the destination directory exists.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the transformer.
            cache_size: The maximum number of cache entries to keep in the cache.
        """
        LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, cache_size)
        self._features: tuple[str, ...] = tuple(features)

    @abstractmethod
    def transform(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Transfigures the specified features in the DataFrame.

        Args:
            dataframe: DataFrame to be transformed.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Returns:
            Transformed DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the specified features in the DataFrame.

        Args:
            dataframe: DataFrame to be transformed.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Returns:
            Transformed DataFrame.
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
