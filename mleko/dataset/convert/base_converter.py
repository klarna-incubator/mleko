"""The module containing the base class for data converter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex

from mleko.cache.format.vaex_cache_format_mixin import VaexCacheFormatMixin
from mleko.cache.lru_cache_mixin import LRUCacheMixin


class BaseConverter(VaexCacheFormatMixin, LRUCacheMixin, ABC):
    """Abstract base class for data converter."""

    def __init__(self, cache_directory: str | Path, cache_size: int):
        """Initialize the `BaseConverter`.

        The `cache_size` is the maximum number of cache entries, and the cache will be cleared if the number of
        entries exceeds this value.

        Args:
            cache_directory: The directory to store the cache in.
            cache_size: The maximum number of cache entries.
        """
        LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, cache_size, False)

    @abstractmethod
    def convert(
        self, file_paths: list[Path] | list[str], cache_group: str | None = None, force_recompute: bool = False
    ) -> vaex.DataFrame:
        """Abstract method to convert the input file paths to the desired output format.

        Args:
            file_paths: A list of input file paths to be converted.
            cache_group: The cache group to use.
            force_recompute: If set to True, forces recomputation and ignores the cache.

        Returns:
            The resulting DataFrame after conversion.
        """
        raise NotImplementedError
