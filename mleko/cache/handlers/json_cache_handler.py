"""This module contains the CacheHandler for serializing and deserializing data using JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base_cache_handler import CacheHandler


def write_json(cache_file_path: Path, output: dict | list) -> None:
    """Writes the given data to a cache file at the specified path, serializing it using JSON.

    Args:
        cache_file_path: A Path object representing the location where the cache file should be saved.
        output: The data to be serialized and saved to the cache file.
    """
    with open(cache_file_path, "w") as cache_file:
        json.dump(output, cache_file, indent=2)


def read_json(cache_file_path: Path) -> Any:
    """Reads the cache file from the specified path and returns the deserialized data.

    Args:
        cache_file_path: A Path object representing the location of the cache file.

    Returns:
        The deserialized data stored in the cache file.
    """
    with open(cache_file_path, "r") as cache_file:
        return json.load(cache_file)


JSON_CACHE_HANDLER = CacheHandler(write_json, read_json, "json", can_handle_none=True)
"""A CacheHandler for serializing and deserializing data using JSON."""
