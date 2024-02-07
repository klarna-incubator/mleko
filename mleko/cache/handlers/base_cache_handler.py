"""This module contains the `CacheHandler` class."""

from pathlib import Path
from typing import Any, Callable, NamedTuple


class CacheHandler(NamedTuple):
    """A named tuple representing a cache handler."""

    writer: Callable[[Path, Any], None]
    """A function that takes a Path object and data as input and saves the data to the cache file."""

    reader: Callable[[Path], Any]
    """A function that takes a Path object as input and returns the deserialized data stored in the cache file."""

    suffix: str
    """The suffix of the cache files."""

    can_handle_none: bool
    """Whether the cache handler can handle None values."""
