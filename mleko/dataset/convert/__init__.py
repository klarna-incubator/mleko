"""The subpackage provides functionality for converting data between different file formats.

The subpackage contains the following converter classes:
    - `BaseConverter`: The abstract base class for all converters.
    - `CSVToVaexConverter`: A converter for converting CSV files to Vaex DataFrames.
"""

from __future__ import annotations

from .base_converter import BaseConverter
from .csv_to_vaex_converter import CSVToVaexConverter


__all__ = ["BaseConverter", "CSVToVaexConverter"]
