"""This module contains the basic `CacheMixin` class for caching the results of method calls.

This class can be used as a mixin to add caching functionality to a class. It provides the basic
functionality for caching the results of method calls based on user-defined cache keys and fingerprints.

The class can be extended to provide additional functionality by inheriting from it and overriding
the `_read_cache_file()` and `_write_cache_file()` methods to customize the cache loading and saving
processes, respectively.

Combining this class with the format mixins can be used to add support for caching different data
formats, such as Vaex DataFrames in Arrow format.
"""
from __future__ import annotations

import hashlib
import inspect
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence

from mleko.cache.fingerprinters.base_fingerprinter import BaseFingerprinter
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level logger instance."""


def get_frame_qualname(frame: inspect.FrameInfo) -> str:
    """Gets the fully qualified name of the function or method associated with the provided frame.

    Args:
        frame: A `FrameInfo` object containing the information of the function or method call.

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


class CacheMixin:
    """A mixin class for caching the results of method calls based on user-defined cache keys and fingerprints.

    The basic functionality of this class is to cache the results of method calls based on user-defined cache keys and
    fingerprints. The cache keys can be a mix of hashable values and tuples containing a value and a BaseFingerprinter
    instance for generating fingerprints. The `CacheMixin` class will save cache files in the specified cache directory
    using the cache key as the filename and the cache file suffix as the file extension. The cache files will be saved
    in the cache directory as pickle files.

    Warning:
        This class maintains an ever-growing cache, which means that the cache size may increase indefinitely
        with new method calls, possibly consuming a large amount of disk space. It does not implement any
        cache eviction strategy. It is recommended to either clear the cache manually when needed or
        use the LRUCacheMixin class, which extends this class to provide an LRU cache mechanism with
        eviction of least recently used cache entries based on a specified maximum number of cache entries.
    """

    def __init__(self, cache_directory: str | Path, cache_file_suffix: str) -> None:
        """Initializes the `CacheMixin` with the provided cache directory.

        Note:
            The cache directory will be created if it does not exist.

        Args:
            cache_directory: The directory where cache files will be stored.
            cache_file_suffix: The suffix/file ending of the cache files.

        Examples:
            >>> from mleko.cache.cache_mixin import CacheMixin
            >>> class MyClass(CacheMixin):
            ...     def __init__(self):
            ...         super().__init__(".cache", "pkl")
            ...
            ...     def my_method(self, x):
            ...         return self._cached_execute(lambda: x ** 2, [x])
            ...
            >>> my_class = MyClass()
            >>> my_class.my_method(2)
            4 # This will be computed and cached
            >>> my_class.my_method(2)
            4 # This will be loaded from the cache
            >>> my_class.my_method(3)
            9 # This will be recomputed and cached
        """
        self._cache_directory = Path(cache_directory)
        self._cache_directory.mkdir(parents=True, exist_ok=True)
        self._cache_file_suffix = cache_file_suffix
        self._cache_type_name = self._find_cache_type_name(self.__class__)

    def _cached_execute(
        self,
        lambda_func: Callable[[], Any],
        cache_keys: list[Hashable | tuple[Any, BaseFingerprinter]],
        force_recompute: bool = False,
    ) -> Any:
        """Executes the given function, caching the results based on the provided cache keys and fingerprints.

        Args:
            lambda_func: A lambda function to execute.
            cache_keys: A list of cache keys that can be a mix of hashable values and tuples containing a value and a
                BaseFingerprinter instance for generating fingerprints.
            force_recompute: A boolean indicating whether to force recompute the result and update the cache, even if a
                cached result is available.

        Returns:
            The result of executing the given function. If a cached result is available and `force_recompute` is False,
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

    def _compute_cache_key(
        self, cache_keys: list[Hashable | tuple[Any, BaseFingerprinter]], frame_qualname: str
    ) -> str:
        """Computes the cache key based on the provided cache keys and the calling function's fully qualified name.

        Args:
            cache_keys: A list of cache keys that can be a mix of hashable values and tuples containing a value and a
                BaseFingerprinter instance for generating fingerprints.
            frame_qualname: The fully qualified name of the cached function stack frame.

        Returns:
            A string representing the computed cache key, which is the MD5 hash of the fully qualified name of the
            calling function or method, along with the fingerprints of the provided cache keys.
        """
        values_to_hash: list[Hashable] = []

        for key in cache_keys:
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], BaseFingerprinter):
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

    def _find_cache_type_name(self, cls: type) -> str | None:
        """Recursively searches the class hierarchy for the name of the class that inherits from `CacheMixin`.

        Args:
            cls: The class to search.

        Returns:
            The name of the class that inherits from `CacheMixin`, or None if no such class exists.
        """
        if CacheMixin.__name__ in cls.__name__:
            return cls.__name__.replace("Mixin", "")

        for base in cls.__bases__:
            found_class_name = self._find_cache_type_name(base)
            if found_class_name:
                return found_class_name
        return None
