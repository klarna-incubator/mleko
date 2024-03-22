"""The module containing the base class for data converter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex

from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema


class BaseConverter(LRUCacheMixin, ABC):
    """Abstract base class for data converter."""

    def __init__(self, cache_directory: str | Path, cache_size: int):
        """Initialize the `BaseConverter`.

        The `cache_size` is the maximum number of cache entries, and the cache will be cleared if the number of
        entries exceeds this value.

        Args:
            cache_directory: The directory to store the cache in.
            cache_size: The maximum number of cache entries.
        """
        LRUCacheMixin.__init__(self, cache_directory, cache_size)

    @abstractmethod
    def convert(
        self,
        file_paths: list[Path] | list[str],
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[DataSchema, vaex.DataFrame]:
        """Abstract method to convert the input file paths to the desired output format.

        Args:
            file_paths: A list of input file paths to be converted.
            cache_group: The cache group to use.
            force_recompute: If set to True, forces recomputation and ignores the cache.
            disable_cache: If set to True, disables the cache.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError
