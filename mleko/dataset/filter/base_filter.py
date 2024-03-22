"""The module containing the base class for data filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex

from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema


class BaseFilter(LRUCacheMixin, ABC):
    """Abstract base class for data filter.

    Will cache the filtered dataframes in the output directory.
    """

    def __init__(self, cache_directory: str | Path, cache_size: int) -> None:
        """Initializes the `BaseFilter` with an output directory.

        Args:
            cache_directory: The target directory where the filtered dataframes are to be saved.
            cache_size: The maximum number of cache entries.
        """
        super().__init__(cache_directory, cache_size)

    @abstractmethod
    def filter(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> vaex.DataFrame:
        """Abstract method to filter the given dataframe.

        Args:
            data_schema: The data schema to be used for filtering.
            dataframe: The dataframe to be filtered.
            cache_group: The cache group to use.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.
            disable_cache: If set to True, disables the cache.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError
