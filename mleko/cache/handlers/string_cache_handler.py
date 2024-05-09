"""This module contains the CacheHandler for serializing and deserializing string data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_cache_handler import CacheHandler


def write_string(cache_file_path: Path, output: str) -> None:
    """Writes the given data to a cache file at the specified path.

    Args:
        cache_file_path: A Path object representing the location where the cache file should be saved.
        output: The data to be serialized and saved to the cache file.
    """
    with open(cache_file_path, "w") as cache_file:
        cache_file.write(output)


def read_string(cache_file_path: Path) -> Any:
    """Reads the cache file from the specified path and returns the deserialized data.

    Args:
        cache_file_path: A Path object representing the location of the cache file.

    Returns:
        The deserialized data stored in the cache file.
    """
    with open(cache_file_path, "r") as cache_file:  # pragma: no cover
        return cache_file.read()


STRING_CACHE_HANDLER = CacheHandler(write_string, read_string, "txt", can_handle_none=False)
"""A CacheHandler for serializing and deserializing string data."""
