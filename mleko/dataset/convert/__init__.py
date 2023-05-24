"""The module provides functionality for converting data between different file formats.

It includes a base class for data converter to support various file format conversions and caching mechanisms.
While it primarily focuses on handling CSV files, the infrastructure allows for extending its
capabilities to other formats as needed.
"""
from __future__ import annotations

from .base_converter import BaseConverter
from .csv_to_vaex_converter import CSVToVaexConverter


__all__ = ["BaseConverter", "CSVToVaexConverter"]
