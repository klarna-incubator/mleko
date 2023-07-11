"""The module containing the mixin class for `vaex` DataFrames to provide HDF5 format caching capabilities."""
from __future__ import annotations

import warnings
from pathlib import Path

import vaex
from tqdm.auto import tqdm

from mleko.utils.tqdm_helpers import set_tqdm_percent_wrapper


class VaexCacheFormatMixin:
    """A mixin class for `vaex` DataFrames to provide HDF5 format caching capabilities.

    This mixin class adds methods for reading and writing HDF5 cache files for `vaex` DataFrames.
    This mixin class is intended to be used with the `Cache` class. It is not intended to be used directly.

    Warning:
        The mixin should be before the cache format class in the inheritance list.

    Examples:
        >>> class MyCacheFormat(VaexCacheFormatMixin, CacheFormat):
        >>>     pass
    """

    _cache_file_suffix = "hdf5"
    """The file extension to use for cache files."""

    def _read_cache_file(self, cache_file_path: Path) -> vaex.DataFrame:
        """Reads a cache file containing a `vaex` DataFrame.

        Args:
            cache_file_path: The path of the cache file to be read.

        Returns:
            The contents of the cache file as a DataFrame.
        """
        return vaex.open(cache_file_path)

    def _write_cache_file(self, cache_file_path: Path, output: vaex.DataFrame) -> None:
        """Writes the results of the DataFrame conversion to HDF5 format in a cache file with `.hdf5` suffix.

        Args:
            cache_file_path: The path of the cache file to be written.
            output: The Vaex DataFrame to be saved in the cache file.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in cast")
            with tqdm(total=100, desc=f"Writing DataFrame to .{self._cache_file_suffix} file") as pbar:
                output.export(
                    cache_file_path,
                    progress=set_tqdm_percent_wrapper(pbar),
                    parallel=True,
                    chunk_size=100_000,
                )
