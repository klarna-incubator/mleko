"""This module contains a LRU cache mixin that can be used by the cache classes.

The `LRUCacheMixin` can be used to add Least Recently Used (LRU) cache functionality to the cached classes.
It evicts the least recently used cache entries when the maximum number of cache entries is exceeded.
The LRU cache mechanism ensures that the most frequently accessed cache entries are retained, while entries that are
rarely accessed and have not been accessed recently are evicted first as the cache fills up. The cache entries
are stored in the cache directory, and the cache is trimmed if needed when cold starting the cache.
"""
from __future__ import annotations

import inspect
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from mleko.utils.custom_logger import CustomLogger

from .cache_mixin import CacheMixin, get_frame_qualname


logger = CustomLogger()
"""A module-level custom logger."""


class LRUCacheMixin(CacheMixin):
    """Least Recently Used Cache Mixin.

    This mixin class extends the CacheMixin to provide a Least Recently Used (LRU) cache mechanism.
    It evicts the least recently used cache entries when the maximum number of cache entries is exceeded.
    The LRU cache mechanism ensures that the most frequently accessed cache entries are retained,
    while entries that are rarely accessed and have not been accessed recently are evicted first as the cache fills up.
    """

    def __init__(self, cache_directory: str | Path, cache_file_suffix: str, cache_size: int) -> None:
        """Initializes the `LRUCacheMixin` with the provided cache directory and maximum number of cache entries.

        Note:
            The cache directory is created if it does not exist. When cold starting the cache, the cache will be loaded
            from the cache directory. The files are sorted by their modification time, and the cache is trimmed if
            needed.

        Args:
            cache_directory: The directory where cache files will be stored.
            cache_file_suffix: The file extension to use for cache files.
            cache_size: The maximum number of cache entries allowed before eviction.

        Examples:
            >>> from mleko.cache import LRUCacheMixin
            >>> class MyClass(LRUCacheMixin):
            ...     def __init__(self):
            ...         super().__init__("cache", "pkl", 2)
            ...
            ...     def my_method(self, x):
            ...         return self._cached_execute(lambda: x ** 2, [x])
            >>> my_class = MyClass()
            >>> my_class.my_method(2)
            4 # This is not cached
            >>> my_class.my_method(2)
            4 # This is cached
            >>> my_class.my_method(3)
            9 # This is not cached
            >>> my_class.my_method(2)
            4 # This is cached
            >>> my_class.my_method(3)
            9 # This is cached
            >>> my_class.my_method(4)
            16 # This is not cached, and the cache is full so the least recently used entry is evicted (x = 2)
        """
        super().__init__(cache_directory, cache_file_suffix)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._load_cache_from_disk()

    def _load_cache_from_disk(self) -> None:
        """Loads the cache entries from the cache directory and initializes the LRU cache.

        Cache entries are ordered by their modification time, and the cache is trimmed if needed.
        """
        frame_qualname = get_frame_qualname(inspect.stack()[2])
        class_name = frame_qualname.split(".")[-2]
        file_name_pattern = rf"{class_name}\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-fA-F\d]{{32}}"
        cache_files = [
            f
            for f in self._cache_directory.glob(f"*.{self._cache_file_suffix}")
            if re.search(file_name_pattern, str(f.stem))
        ]
        ordered_cache_files = sorted(cache_files, key=lambda x: x.stat().st_mtime)
        for cache_file in ordered_cache_files:
            cache_key_match = re.search(file_name_pattern, cache_file.stem)
            cache_key = cache_key_match.group(0)  # type: ignore
            if cache_key not in self._cache:
                if len(self._cache) >= self._cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    for file in self._cache_directory.glob(f"{oldest_key}*.{self._cache_file_suffix}"):
                        file.unlink()
                self._cache[cache_key] = True

    def _load_from_cache(self, cache_key: str) -> Any | None:
        """Loads data from the cache based on the provided cache key and updates the LRU cache.

        Args:
            cache_key: A string representing the cache key.

        Returns:
            The cached data if it exists, or None if there is no data for the given cache key.
        """
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
        return super()._load_from_cache(cache_key)

    def _save_to_cache(self, cache_key: str, output: Any) -> None:
        """Saves the given data to the cache using the provided cache key, updating the LRU cache accordingly.

        If the cache reaches its maximum size, the least recently used entry will be evicted.

        Args:
            cache_key: A string representing the cache key.
            output: The data to be saved to the cache.
        """
        if cache_key not in self._cache:
            if len(self._cache) >= self._cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                for file in self._cache_directory.glob(f"{oldest_key}*.{self._cache_file_suffix}"):
                    file.unlink()
            self._cache[cache_key] = True
        else:
            self._cache.move_to_end(cache_key)
        super()._save_to_cache(cache_key, output)
