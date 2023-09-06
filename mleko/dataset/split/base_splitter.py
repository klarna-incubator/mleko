"""The module containing the base class for data splitter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex

from mleko.cache.lru_cache_mixin import LRUCacheMixin


class BaseSplitter(LRUCacheMixin, ABC):
    """Abstract base class for data splitter.

    Will cache the split dataframes in the output directory.
    """

    def __init__(self, cache_directory: str | Path, cache_size: int):
        """Initializes the `BaseSplitter` with an output directory.

        Args:
            cache_directory: The target directory where the split dataframes are to be saved.
            cache_size: The maximum number of cache entries.
        """
        super().__init__(cache_directory, cache_size)

    @abstractmethod
    def split(
        self, dataframe: vaex.DataFrame, cache_group: str | None = None, force_recompute: bool = False
    ) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Abstract method to split the given dataframe into two parts.

        Args:
            dataframe: The dataframe to be split.
            cache_group: The cache group to use.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError
