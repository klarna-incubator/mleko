"""This module contains cache format mixins which can be used to add support for caching different file formats.

The following cache format mixins are provided:
    - `VaexCacheFormatMixin`: A mixin that adds support for caching Vaex DataFrames in Arrow format.
"""
from __future__ import annotations

from .vaex_cache_format_mixin import VaexCacheFormatMixin


__all__ = ["VaexCacheFormatMixin"]
