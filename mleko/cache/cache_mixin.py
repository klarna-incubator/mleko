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
        disable_cache: bool = False,
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
            disable_cache: Overrides the class-level `disable_cache` attribute. If set to True, disables the cache.

        Returns:
            A tuple containing a boolean indicating whether the cached result was used, and the result of executing the
            given function. If a cached result is available and `force_recompute` is False, the cached result will be
            returned instead of recomputing the result.
        """
        if self._disable_cache or disable_cache:
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
            msg = (
                f"The computed cache key is too long ({len(cache_key)} chars)."
                "The maximum length of a cache key is 235 chars, and given the current class, the maximum "
                f"length of the provided cache_group is {235 - len(cache_key)} chars. "
                "Please reduce the length of the cache_group."
            )
            logger.error(msg)
            raise ValueError(msg)

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
        ).union({f".{PICKLE_CACHE_HANDLER.suffix}"})

        cache_file_paths = [
            f
            for f in sorted(list(self._cache_directory.glob(f"{cache_key}*.*")), key=extract_number)
            if f.suffix in cache_handler_suffixes
        ]

        if cache_file_paths:
            output_data = []
            for i, cache_file_path in enumerate(cache_file_paths):
                handler = self._get_handler(cache_handlers, i)
                if cache_file_path.suffix != f".{handler.suffix}":
                    handler = PICKLE_CACHE_HANDLER

                output_data.append(handler.reader(cache_file_path))
            return tuple(output_data) if len(output_data) > 1 else output_data[0]
        return None

    def _get_handler(self, cache_handlers: CacheHandler | list[CacheHandler], index: int = 0) -> CacheHandler:
        """Gets the cache handler at the given index.

        Args:
            cache_handlers: A CacheHandler instance or a list of CacheHandler instances.
            index: The index of the cache handler to get.

        Returns:
            Handler at the given index. If a single CacheHandler instance is provided, it will be returned.
        """
        if isinstance(cache_handlers, list):
            return cache_handlers[index]
        return cache_handlers

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
            for i, output_item in enumerate(output):
                self._write_to_cache_file(cache_key, output_item, i, cache_handlers, is_sequence_output=True)
        else:
            self._write_to_cache_file(cache_key, output, 0, cache_handlers, is_sequence_output=False)

    def _write_to_cache_file(
        self,
        cache_key: str,
        output_item: Any,
        index: int,
        cache_handlers: CacheHandler | list[CacheHandler],
        is_sequence_output: bool,
    ) -> None:
        """Writes the given data to the cache file using the provided cache key.

        If the output is None and the cache handler cannot handle None, the output will be saved using the pickle
        cache handler. Otherwise, the output will be saved to a cache file using the provided cache handler.

        Args:
            cache_key: A string representing the cache key.
            output_item: The data to be saved to the cache.
            index: The index of the cache handler to use.
            cache_handlers: A CacheHandler instance or a list of CacheHandler instances.
            is_sequence_output: Whether the output is a sequence or not. If True, the cache file will be saved with the
                index appended to the cache key.
        """
        handler = self._get_handler(cache_handlers, index)

        file_suffix = f"_{index}" if is_sequence_output else ""
        if output_item is None and not handler.can_handle_none:
            cache_file_path = self._cache_directory / f"{cache_key}{file_suffix}.{PICKLE_CACHE_HANDLER.suffix}"
            PICKLE_CACHE_HANDLER.writer(cache_file_path, output_item)
        else:
            cache_file_path = self._cache_directory / f"{cache_key}{file_suffix}.{handler.suffix}"
            handler.writer(cache_file_path, output_item)

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
