"""The subpackage contains the core data processing functionality of the `MLEKO` library.

This subpackage focuses on handling different aspects of data processing, including fetching data from various
sources, converting between file formats, filtering, and feature engineering.

The following submodules are provided:
    - `ingest`: The submodule provides functionality for fetching data from various sources.
    - `convert`: The submodule provides functionality for converting between different file formats.
    - `split`: The submodule provides functionality for splitting data into multiple parts.
    - `feature_select`: The submodule provides functionality for selecting features from data.
"""

from __future__ import annotations

from .data_schema import DataSchema


__all__ = ["DataSchema"]
