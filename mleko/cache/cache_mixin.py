"""This module contains the basic `CacheMixin` class for caching the results of method calls.

This class can be used as a mixin to add caching functionality to a class. It provides the basic
functionality for caching the results of method calls based on user-defined cache keys and fingerprints.

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
from mleko.cache.handlers import PICKLE_CACHE_HANDLER, CacheHandler
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level logger instance."""


def get_qualified_name_from_frame(frame: inspect.FrameInfo) -> str:
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


def get_qualified_name_of_caller(frame_depth: int) -> str:
    """Gets the fully qualified name of the calling function or method.

    The fully qualified name is in the format "module.class.method" for class methods or "module.function" for
    functions.

    Args:
        frame_depth: The depth of the frame to inspect. The default value is 2, which is the frame of the calling
            function or method. For each nested function or method, the frame depth should be increased by 1.

    Returns:
        A string representing the fully qualified name of the calling function or method.
    """
    frame_qualname = get_qualified_name_from_frame(inspect.stack()[frame_depth])
    class_method_name = ".".join(frame_qualname.split(".")[-2:])
    return class_method_name


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

    def __init__(self, cache_directory: str | Path, disable_cache: bool) -> None:
        """Initializes the `CacheMixin` with the provided cache directory.

        Note:
            The cache directory will be created if it does not exist.

        Args:
            cache_directory: The directory where cache files will be stored.
            disable_cache: Whether to disable the cache.

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
        self._cache_type_name = self._find_cache_type_name(self.__class__)
        self._disable_cache = disable_cache

    def _cached_execute(
        self,
        lambda_func: Callable[[], Any],
        cache_key_inputs: list[Hashable | tuple[Any, BaseFingerprinter]],
        cache_group: str | None = None,
        force_recompute: bool = False,
        cache_handlers: CacheHandler | list[CacheHandler] | None = None,
    ) -> Any:
        """Executes the given function, caching the results based on the provided cache keys and fingerprints.

        Warning:
            The cache group is used to group related cache keys together to prevent collisions between cache keys
            originating from the same method. For example, if a method is called during the training and testing
            phases of a machine learning pipeline, the cache keys for the training and testing phases should be
            using different cache groups to prevent collisions between the cache keys for the two phases. Otherwise,
            the later cache keys might overwrite the earlier cache entries.

        Args:
            lambda_func: A lambda function to execute.
            cache_key_inputs: A list of cache keys that can be a mix of hashable values and tuples containing
                a value and a BaseFingerprinter instance for generating fingerprints.
            cache_group: A string representing the cache group, used to group related cache keys together when methods
                are called independently.
            force_recompute: A boolean indicating whether to force recompute the result and update the cache, even if a
                cached result is available.
            cache_handlers: A CacheHandler instance or a list of CacheHandler instances. If None, the cache files will
                be read using pickle. If a single CacheHandler instance is provided, it will be used for all cache
                files. If a list of CacheHandler instances is provided, each CacheHandler instance will be used for
                each cache file.

        Returns:
            A tuple containing a boolean indicating whether the cached result was used, and the result of executing the
            given function. If a cached result is available and `force_recompute` is False, the cached result will be
            returned instead of recomputing the result.
        """
        if self._disable_cache:
            return lambda_func()

        if cache_handlers is None:
            cache_handlers = PICKLE_CACHE_HANDLER

        class_method_name = get_qualified_name_of_caller(2)
        cache_key = self._compute_cache_key(cache_key_inputs, cache_group)

        if not force_recompute:
            output = self._load_from_cache(cache_key, cache_handlers)
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
        self._save_to_cache(cache_key, output, cache_handlers)
        return self._load_from_cache(cache_key, cache_handlers)

    def _compute_cache_key(
        self,
        cache_key_inputs: list[Hashable | tuple[Any, BaseFingerprinter]],
        cache_group: str | None = None,
        frame_depth: int = 3,
    ) -> str:
        """Computes the cache key based on the provided cache keys and the calling function's fully qualified name.

        Args:
            cache_key_inputs: A list of cache keys that can be a mix of hashable values and tuples containing a
                value and a BaseFingerprinter instance for generating fingerprints.
            cache_group: A string representing the cache group.
            frame_depth: The depth of the frame to inspect. The default value is 2, which is the frame of the calling
                function or method. For each nested function or method, the frame depth should be increased by 1.

        Raises:
            ValueError: If the computed cache key is too long.

        Returns:
            A string representing the computed cache key, which is the MD5 hash of the fully qualified name of the
            calling function or method, along with the fingerprints of the provided cache keys.
        """
        values_to_hash: list[Hashable] = []

        for key_input in cache_key_inputs:
            if isinstance(key_input, tuple) and len(key_input) == 2 and isinstance(key_input[1], BaseFingerprinter):
                value, fingerprinter = key_input
                values_to_hash.append(fingerprinter.fingerprint(value))
            else:
                values_to_hash.append(key_input)

        logger.debug(f"Cache key inputs: {values_to_hash}")
        data = pickle.dumps(values_to_hash)
        cache_key_prefix = get_qualified_name_of_caller(frame_depth)
        if cache_group is not None:
            cache_key_prefix = f"{cache_key_prefix}.{cache_group}"

        cache_key = f"{cache_key_prefix}.{hashlib.md5(data).hexdigest()}"
        if len(cache_key) > 235:
            raise ValueError(
                f"The computed cache key is too long ({len(cache_key)} chars)."
                "The maximum length of a cache key is 235 chars, and given the current class, the maximum "
                f"length of the provided cache_group is {235 - len(cache_key)} chars. "
                "Please reduce the length of the cache_group."
            )

        return f"{cache_key_prefix}.{hashlib.md5(data).hexdigest()}"

    def _load_from_cache(
        self,
        cache_key: str,
        cache_handlers: CacheHandler | list[CacheHandler],
    ) -> Any | None:
        """Loads data from the cache based on the provided cache key.

        Args:
            cache_key: A string representing the cache key.
            cache_handlers: A CacheHandler instance or a list of CacheHandler instances. If a single CacheHandler
                instance is provided, it will be used for all cache files. If a list of CacheHandler instances is
                provided, each CacheHandler instance will be used for each cache file.

        Returns:
            The cached data if it exists, or None if there is no data for the given cache key.
        """

        def extract_number(file_path: Path) -> int:
            result = re.search(r"[a-fA-F\d]{32}_(\d+).", str(file_path))
            return int(result.group(1)) if result else 0

        cache_handler_suffixes = (
            {f".{cache_handler.suffix}" for cache_handler in cache_handlers}
            if isinstance(cache_handlers, list)
            else {f".{cache_handlers.suffix}"}
        )
        cache_file_paths = [
            f
            for f in sorted(list(self._cache_directory.glob(f"{cache_key}*.*")), key=extract_number)
            if f.suffix in cache_handler_suffixes
        ]
        if cache_file_paths:
            output_data = []
            for i, cache_file_path in enumerate(cache_file_paths):
                reader = cache_handlers[i].reader if isinstance(cache_handlers, list) else cache_handlers.reader
                output_data.append(reader(cache_file_path))
            return tuple(output_data) if len(output_data) > 1 else output_data[0]
        return None

    def _save_to_cache(
        self,
        cache_key: str,
        output: Any | Sequence[Any],
        cache_handlers: CacheHandler | list[CacheHandler],
    ) -> None:
        """Saves the given data to the cache using the provided cache key.

        If the output is a sequence, each element will be saved to a separate cache file. Otherwise, the output will be
        saved to a single cache file. The cache file will be saved in the cache directory with the cache key as the
        filename and the cache file suffix as the file extension.

        Args:
            cache_key: A string representing the cache key.
            output: The data to be saved to the cache.
            cache_handlers: A CacheHandler instance or a list of CacheHandler instances. If a single CacheHandler
                instance is provided, it will be used for all cache files. If a list of CacheHandler instances is
                provided, each CacheHandler instance will be used for each cache file.
        """
        if isinstance(output, Sequence):
            for i in range(len(output)):
                writer = cache_handlers[i].writer if isinstance(cache_handlers, list) else cache_handlers.writer
                suffix = cache_handlers[i].suffix if isinstance(cache_handlers, list) else cache_handlers.suffix
                cache_file_path = self._cache_directory / f"{cache_key}_{i}.{suffix}"
                writer(cache_file_path, output[i])
        else:
            writer = cache_handlers[0].writer if isinstance(cache_handlers, list) else cache_handlers.writer
            suffix = cache_handlers[0].suffix if isinstance(cache_handlers, list) else cache_handlers.suffix
            cache_file_path = self._cache_directory / f"{cache_key}.{suffix}"
            writer(cache_file_path, output)

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
        return None  # pragma: no cover
