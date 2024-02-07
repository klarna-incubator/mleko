"""This module contains the CacheHandler for serializing and deserializing data using pickle."""

import pickle
from pathlib import Path
from typing import Any

from .base_cache_handler import CacheHandler


def write_pickle(cache_file_path: Path, output: Any) -> None:
    """Writes the given data to a cache file at the specified path, serializing it using pickle.

    Args:
        cache_file_path: A Path object representing the location where the cache file should be saved.
        output: The data to be serialized and saved to the cache file.
    """
    with open(cache_file_path, "wb") as cache_file:
        pickle.dump(output, cache_file)


def read_pickle(cache_file_path: Path) -> Any:
    """Reads the cache file from the specified path and returns the deserialized data.

    Args:
        cache_file_path: A Path object representing the location of the cache file.

    Returns:
        The deserialized data stored in the cache file.
    """
    with open(cache_file_path, "rb") as cache_file:
        return pickle.load(cache_file)


PICKLE_CACHE_HANDLER = CacheHandler(write_pickle, read_pickle, "pkl", can_handle_none=True)
"""A CacheHandler for pickling Python objects."""
