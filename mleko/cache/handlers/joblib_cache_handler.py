"""This module contains the CacheHandler for serializing and deserializing data using joblib."""

from pathlib import Path
from typing import Any

import joblib

from .base_cache_handler import CacheHandler


def write_joblib(cache_file_path: Path, output: Any) -> None:
    """Writes the given data to a cache file at the specified path, serializing it using joblib.

    Args:
        cache_file_path: A Path object representing the location where the cache file should be saved.
        output: The data to be serialized and saved to the cache file.
    """
    joblib.dump(output, cache_file_path)


def read_joblib(cache_file_path: Path) -> Any:
    """Reads the cache file from the specified path and returns the deserialized data.

    Args:
        cache_file_path: A Path object representing the location of the cache file.

    Returns:
        The deserialized data stored in the cache file.
    """
    return joblib.load(cache_file_path)


JOBLIB_CACHE_HANDLER = CacheHandler(write_joblib, read_joblib, "joblib", can_handle_none=True)
"""A CacheHandler for Python objects using joblib."""
