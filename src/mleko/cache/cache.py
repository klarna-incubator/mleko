"""A mixin module that provides a flexible and configurable caching mechanism for method call results.

This module contains mixin classes designed to cache method call results efficiently, reducing
the overhead of repetitive calculations or data fetching. The CacheMixin class stores the results
of method calls on a per-instance basis, based on user-defined cache keys and fingerprints. The
LRUCacheMixin class extends the CacheMixin to implement a Least Recently Used (LRU) cache eviction
mechanism that helps manage the cache size by evicting the least recently used cache entries when
the specified maximum number of cache entries is exceeded.

Usage:
    CacheMixin should be subclassed by other classes that require caching capabilities for their method calls.
    The subclass should implement the method logic inside a lambda function and pass it to the `_cached_execute`
    method, along with any cache keys needed to identify unique results. For more advanced use cases,
    the LRUCacheMixin can be employed to enable automatic cache eviction based on an LRU strategy.
"""
from __future__ import annotations

import hashlib
import inspect
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence

import vaex
from tqdm import tqdm

from mleko.cache.fingerprinters import Fingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.tqdm import set_tqdm_percent_wrapper


logger = CustomLogger()
"""A CustomLogger instance that's used throughout the module for logging."""


def get_frame_qualname(frame: inspect.FrameInfo) -> str:
    """Gets the fully qualified name of the function or method associated with the provided frame.

    Args:
        frame: A FrameInfo object containing the information of the function or method call.

    Returns:
        A string representing the fully qualified name, in the format "module.class.method" for class methods or
        "module.function" for functions.
    """
    caller_function = frame.function
    caller_obj = inspect.getmodule(frame[0])
    module_name = caller_obj.__name__ if caller_obj is not None else "__main__"
    if "self" in frame.frame.f_locals:
        class_name = frame.frame.f_locals["self"].__class__.__name__
        return f"{module_name}.{class_name}.{caller_function}"
    return f"{module_name}.{caller_function}"


class VaexArrowCacheFormatMixin:
    """A mixin class for Vaex DataFrames to provide Arrow format caching capabilities.

    This mixin class adds methods for reading and writing arrow cache files for Vaex DataFrames.
    """

    cache_file_suffix = "arrow"
    """The file extension to use for cache files."""

    def _read_cache_file(self, cache_file_path: Path) -> vaex.DataFrame:
        """Reads a cache file containing a Vaex DataFrame.

        Args:
            cache_file_path: The path of the cache file to be read.

        Returns:
            The contents of the cache file as a DataFrame.
        """
        return vaex.open(cache_file_path)

    def _write_cache_file(self, cache_file_path: Path, output: vaex.DataFrame) -> None:
        """Writes the results of the DataFrame conversion to Arrow format in a cache file with arrow suffix.

        Args:
            cache_file_path: The path of the cache file to be written.
            output: The Vaex DataFrame to be saved in the cache file.
        """
        with tqdm(total=100, desc="Writing DataFrame to Arrow file") as pbar:
            output.export_arrow(
                cache_file_path,
                progress=set_tqdm_percent_wrapper(pbar),
                parallel=True,
                reduce_large=True,
            )
        output.close()


class CacheMixin:
    """A mixin class for caching the results of method calls based on user-defined cache keys and fingerprints.

    Warning:
        This class maintains an ever-growing cache, which means that the cache size may increase indefinitely
        with new method calls, possibly consuming a large amount of disk space. It does not implement any
        cache eviction strategy. It is recommended to either clear the cache manually when needed or
        use the LRUCacheMixin class, which extends this class to provide an LRU cache mechanism with
        eviction of least recently used cache entries based on a specified maximum number of cache entries.
    """

    def __init__(self, cache_directory: str | Path, cache_file_suffix: str) -> None:
        """Initializes the CacheMixin with the provided cache directory.

        Args:
            cache_directory: The directory where cache files will be stored.
            cache_file_suffix: The suffix/file ending of the cache files.
        """
        self._cache_directory = Path(cache_directory)
        self._cache_directory.mkdir(parents=True, exist_ok=True)
        self._cache_file_suffix = cache_file_suffix
        self._cache_type_name = [
            base.__name__ for base in self.__class__.__bases__ if CacheMixin.__name__ in base.__name__
        ][0].replace("Mixin", "")

    def _cached_execute(
        self,
        lambda_func: Callable[[], Any],
        cache_keys: list[Hashable | tuple[Any, Fingerprinter]],
        force_recompute: bool = False,
    ) -> Any:
        """Executes the given function, caching the results based on the provided cache keys and fingerprints.

        Args:
            lambda_func: A lambda function to execute.
            cache_keys: A list of cache keys that can be a mix of hashable values and tuples containing a value and a
                Fingerprinter instance for generating fingerprints.
            force_recompute: A boolean indicating whether to force recompute the result and update the cache, even if a
                cached result is available.

        Returns:
            The result of executing the given function. If a cached result is available and force_recompute is False,
            the cached result will be returned instead of recomputing the result.
        """
        frame_qualname = get_frame_qualname(inspect.stack()[1])
        class_method_name = ".".join(frame_qualname.split(".")[-2:])
        cache_key = self._compute_cache_key(cache_keys, frame_qualname)

        if not force_recompute:
            output = self._load_from_cache(cache_key)
            if output is not None:
                logger.info(
                    f"\033[32mCache Hit\033[0m ({self._cache_type_name}) {class_method_name}: Using cached output."
                )
                return output
            else:
                logger.info(
                    f"\033[31mCache Miss\033[0m ({self._cache_type_name}) {class_method_name}: Executing method."
                )
        else:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m ({self._cache_type_name}) {class_method_name}: Executing method."
            )

        output = lambda_func()
        self._save_to_cache(cache_key, output)
        return self._load_from_cache(cache_key)

    def _compute_cache_key(self, cache_keys: list[Hashable | tuple[Any, Fingerprinter]], frame_qualname: str) -> str:
        """Computes the cache key based on the provided cache keys and the calling function's fully qualified name.

        Args:
            cache_keys: A list of cache keys that can be a mix of hashable values and tuples containing a value and a
                Fingerprinter instance for generating fingerprints.
            frame_qualname: The fully qualified name of the cached function stack frame.

        Returns:
            A string representing the computed cache key, which is the MD5 hash of the fully qualified name of the
            calling function or method, along with the fingerprints of the provided cache keys.
        """
        values_to_hash: list[Hashable] = []

        for key in cache_keys:
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], Fingerprinter):
                value, fingerprinter = key
                values_to_hash.append(fingerprinter.fingerprint(value))
            else:
                values_to_hash.append(key)

        data = pickle.dumps((frame_qualname, values_to_hash))

        class_method_name = ".".join(frame_qualname.split(".")[-2:])
        cache_key = f"{class_method_name}.{hashlib.md5(data).hexdigest()}"

        return cache_key

    def _read_cache_file(self, cache_file_path: Path) -> Any:
        """Reads the cache file from the specified path and returns the deserialized data.

        This method can be overridden in subclasses to customize the cache loading process.

        Args:
            cache_file_path: A Path object representing the location of the cache file.

        Returns:
            The deserialized data stored in the cache file.
        """
        with open(cache_file_path, "rb") as cache_file:
            return pickle.load(cache_file)

    def _load_from_cache(self, cache_key: str) -> Any | None:
        """Loads data from the cache based on the provided cache key.

        Args:
            cache_key: A string representing the cache key.

        Returns:
            The cached data if it exists, or None if there is no data for the given cache key.
        """

        def extract_number(file_path: Path) -> int:
            result = re.search(r"[a-fA-F\d]{32}_(\d+).", str(file_path))
            return int(result.group(1)) if result else 0

        cache_file_paths = sorted(
            list(self._cache_directory.glob(f"{cache_key}*.{self._cache_file_suffix}")), key=extract_number
        )
        if cache_file_paths:
            output_data = []
            for cache_file_path in cache_file_paths:
                output_data.append(self._read_cache_file(cache_file_path))
            return tuple(output_data) if len(output_data) > 1 else output_data[0]
        return None

    def _write_cache_file(self, cache_file_path: Path, output: Any) -> None:
        """Writes the given data to a cache file at the specified path, serializing it using pickle.

        This method can be overridden in subclasses to customize the cache saving process.

        Args:
            cache_file_path: A Path object representing the location where the cache file should be saved.
            output: The data to be serialized and saved to the cache file.
        """
        with open(cache_file_path, "wb") as cache_file:
            pickle.dump(output, cache_file)

    def _save_to_cache(self, cache_key: str, output: Any | Sequence[Any]) -> None:
        """Saves the given data to the cache using the provided cache key.

        If the output is a sequence, each element will be saved to a separate cache file. Otherwise, the output will be
        saved to a single cache file. The cache file will be saved in the cache directory with the cache key as the
        filename and the cache file suffix as the file extension.

        Args:
            cache_key: A string representing the cache key.
            output: The data to be saved to the cache.
        """
        if isinstance(output, Sequence):
            for i in range(len(output)):
                cache_file_path = self._cache_directory / f"{cache_key}_{i}.{self._cache_file_suffix}"
                self._write_cache_file(cache_file_path, output[i])
        else:
            cache_file_path = self._cache_directory / f"{cache_key}.{self._cache_file_suffix}"
            self._write_cache_file(cache_file_path, output)


class LRUCacheMixin(CacheMixin):
    """Least Recently Used Cache Mixin.

    This mixin class extends the CacheMixin to provide a Least Recently Used (LRU) cache mechanism.
    It evicts the least recently used cache entries when the maximum number of cache entries is exceeded.
    The LRU cache mechanism ensures that the most frequently accessed cache entries are retained,
    while entries that are rarely accessed and have not been accessed recently are evicted first as the cache fills up.
    """

    def __init__(self, cache_directory: str | Path, cache_file_suffix: str, max_entries: int) -> None:
        """Initializes the LRUCacheMixin with the provided cache directory and maximum number of cache entries.

        Args:
            cache_directory: The directory where cache files will be stored.
            cache_file_suffix: The file extension to use for cache files.
            max_entries: The maximum number of cache entries allowed before eviction.
        """
        super().__init__(cache_directory, cache_file_suffix)
        self._max_entries = max_entries
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._load_cache_from_disk()

    def _load_cache_from_disk(self) -> None:
        """Loads the cache entries from the cache directory and initializes the LRU cache.

        Cache entries are ordered by their modification time, and the cache is trimmed if needed.
        """
        frame_qualname = get_frame_qualname(inspect.stack()[2])
        class_name = frame_qualname.split(".")[-2]
        cache_files = [
            f
            for f in self._cache_directory.glob(f"*.{self._cache_file_suffix}")
            if re.search(rf"{class_name}\.[a-zA-Z]+\.[a-fA-F\d]{{32}}", str(f.stem))
        ]
        ordered_cache_files = sorted(cache_files, key=lambda x: x.stat().st_mtime)
        for cache_file in ordered_cache_files:
            cache_key = cache_file.stem.split("_")[0]
            if cache_key not in self._cache:
                if len(self._cache) >= self._max_entries:
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
            if len(self._cache) >= self._max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                for file in self._cache_directory.glob(f"{oldest_key}*.{self._cache_file_suffix}"):
                    file.unlink()
            self._cache[cache_key] = True
        else:
            self._cache.move_to_end(cache_key)
        super()._save_to_cache(cache_key, output)
