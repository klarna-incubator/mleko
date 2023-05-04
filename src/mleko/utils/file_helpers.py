"""Utility functions for file system operations.

This module provides various utility functions for working with files and directories,
such as reading data from a JSON file and clearing a directory of its contents.

Functions:
- clear_directory: Deletes all files inside the directory matching the pattern.
"""
from __future__ import annotations

from pathlib import Path


def clear_directory(directory: Path, pattern: str = "*") -> None:
    """Deletes all files inside the directory matching the pattern.

    Args:
        directory: Directory to be cleaned.
        pattern: Glob pattern that includes target files. Defaults to "*".
    """
    for f in directory.glob(pattern):
        f.unlink()
