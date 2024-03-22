"""This module contains the CacheHandler for reading a writing `vaex` DataFrames to disk."""

import warnings
from pathlib import Path

import vaex
from tqdm.auto import tqdm

from mleko.utils.tqdm_helpers import set_tqdm_percent_wrapper

from .base_cache_handler import CacheHandler


def read_vaex_dataframe(cache_file_path: Path) -> vaex.DataFrame:
    """Reads a cache file containing a `vaex` DataFrame.

    Args:
        cache_file_path: The path of the cache file to be read.

    Returns:
        The contents of the cache file as a DataFrame.
    """
    return vaex.open(cache_file_path)


def write_vaex_dataframe(cache_file_path: Path, output: vaex.DataFrame) -> None:
    """Writes the results of the DataFrame conversion to a file.

    Args:
        cache_file_path: The path of the cache file to be written.
        output: The Vaex DataFrame to be saved in the cache file.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in cast")
        with tqdm(total=100, desc=f"Writing DataFrame to {cache_file_path.suffix} file") as pbar:
            output.export_hdf5(
                cache_file_path,
                progress=set_tqdm_percent_wrapper(pbar),
                parallel=True,
                chunk_size=100_000,
            )


VAEX_DATAFRAME_CACHE_HANDLER = CacheHandler(
    writer=write_vaex_dataframe, reader=read_vaex_dataframe, suffix="hdf5", can_handle_none=False
)
"""A CacheHandler for `vaex` DataFrames."""
