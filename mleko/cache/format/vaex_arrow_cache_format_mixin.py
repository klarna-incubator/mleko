"""The module containing the mixin class for Vaex DataFrames to provide Arrow format caching capabilities."""
from __future__ import annotations

from pathlib import Path

import vaex
from mleko.utils.tqdm_helpers import set_tqdm_percent_wrapper
from tqdm import tqdm


class VaexArrowCacheFormatMixin:
    """A mixin class for Vaex DataFrames to provide Arrow format caching capabilities.

    This mixin class adds methods for reading and writing arrow cache files for Vaex DataFrames.

    Note:
        This mixin class is intended to be used with the `Cache` class. It is not intended to be used
        directly.

    Warning:
        The mixin should be before the cache format class in the inheritance list.

    Examples:
        >>> class MyCacheFormat(VaexArrowCacheFormatMixin, CacheFormat):
        >>>     pass
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
