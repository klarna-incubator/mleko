"""Subpackage with utility functions and classes.

This subpackage contains utility functions and classes that are used throughout the project. These include a custom
logger, decorators, and helper functions for working with `vaex` DataFrames and `tqdm` progress bars.
"""

from __future__ import annotations

from .custom_logger import CustomLogger
from .decorators import auto_repr, timing
from .file_helpers import LocalFileEntry, LocalManifest, LocalManifestHandler, clear_directory
from .s3_helpers import S3Client, S3FileManifest
from .tqdm_helpers import set_tqdm_percent_wrapper
from .vaex_helpers import get_column, get_columns, get_filtered_df, get_indices


__all__ = [
    "CustomLogger",
    "auto_repr",
    "timing",
    "clear_directory",
    "LocalFileEntry",
    "LocalManifest",
    "LocalManifestHandler",
    "set_tqdm_percent_wrapper",
    "get_column",
    "get_columns",
    "get_filtered_df",
    "get_indices",
    "S3Client",
    "S3FileManifest",
]
