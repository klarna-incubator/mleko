"""This module provides utility functions for file and directory operations."""

from __future__ import annotations

from pathlib import Path


def clear_directory(directory: Path, pattern: str = "*") -> None:
    """Remove all files in a directory that match a given pattern.

    This function takes a directory and, using the provided pattern, searches for all matching files
    and removes them. This is useful when cleaning up temporary or intermediate files in a workspace.

    Args:
        directory: The `Path` object referring to the directory to be cleared.
        pattern: The search pattern to match the files in the directory (default: "*", matches all files).
    """
    for f in directory.glob(pattern):
        f.unlink()
