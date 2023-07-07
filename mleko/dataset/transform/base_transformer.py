"""Module for the base transformer class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import joblib
import vaex

from mleko.cache.fingerprinters.base_fingerprinter import BaseFingerprinter
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
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
        disable_cache: bool,
    ) -> None:
        """Initializes the transformer and ensures the destination directory exists.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the transformer.
            cache_size: The maximum number of cache entries to keep in the cache.
            disable_cache: Whether to disable the cache.
        """
        LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, cache_size, disable_cache)
        self._features: tuple[str, ...] = tuple(features)
        self._transformer = None

    def transform(
        self, dataframe: vaex.DataFrame, fit: bool, cache_group: str | None = None, force_recompute: bool = False
    ) -> vaex.DataFrame:
        """Transfigures the specified features in the DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame to be transformed.
            fit: Whether to fit the transformer on the input data.
            cache_group: The cache group to use.
            force_recompute: Whether to force the transformation to be recomputed even if the result is cached.

        Returns:
            Transformed DataFrame.
        """
        cache_keys = [self._fingerprint(), (dataframe, VaexFingerprinter())]
        cached, df = self._cached_execute(
            lambda_func=lambda: self._transform(dataframe, fit),
            cache_keys=cache_keys,
            cache_group=cache_group,
            force_recompute=force_recompute,
        )

        if fit and not self._disable_cache:
            self._save_or_load_transformer(cached, cache_group, cache_keys)

        return df

    @abstractmethod
    def _transform(self, dataframe: vaex.DataFrame, fit: bool) -> vaex.DataFrame:
        """Transfigures the specified features in the DataFrame.

        Args:
            dataframe: DataFrame to be transformed.
            fit: Whether to fit the transformer on the input data.

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

    def _save_or_load_transformer(
        self, load: bool, cache_group: str | None, cache_keys: list[Hashable | tuple[Any, BaseFingerprinter]]
    ) -> None:
        """Saves or loads the transformer to/from the cache.

        Will save the transformer to the cache if `load` is False, otherwise will load the transformer from the cache.
        The cache key is computed using the specified cache keys and will be used to save the transformer to the cache
        using joblib.

        Args:
            load: Whether to load the transformer from the cache or save it to the cache.
            cache_group: The cache group to use.
            cache_keys: The cache keys to use for the transformer.
        """
        cache_key = self._compute_cache_key(cache_keys, cache_group, frame_depth=3)
        transformer_path = self._cache_directory / f"{cache_key}.transformer"

        if load:
            logger.info(f"Loading transformer from {transformer_path}.")
            self._transformer = joblib.load(transformer_path)
        else:
            logger.info(f"Saving transformer to {transformer_path}.")
            joblib.dump(self._transformer, transformer_path)
