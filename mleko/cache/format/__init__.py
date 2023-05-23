"""This module contains cache format mixins usable by the cache classes.

These are optional but can be used to add functionality to the cache classes. For example, the
`VaexArrowCacheFormatMixin` mixin can be used to add support for caching Vaex DataFrames in Arrow
format.
"""
from __future__ import annotations

from .vaex_arrow_cache_format_mixin import VaexArrowCacheFormatMixin


__all__ = ["VaexArrowCacheFormatMixin"]
