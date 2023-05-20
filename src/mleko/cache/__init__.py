"""A subpackage providing caching mixins and fingerprinting utilities to efficiently cache method call results.

This subpackage contains two main components:

1. Caching Mixins: A flexible caching mechanism for method call results. These mixin classes allow efficient
   caching and retrieval of method call results, reducing overhead from repetitive calculations or data fetching.
   The CacheMixin class provides a basic cache implementation, while LRUCacheMixin extends it with a Least
   Recently Used (LRU) cache eviction strategy, allowing better cache size management.

2. Fingerprinters: A set of Fingerprinter classes designed to generate unique fingerprints of various data
   and file types such as Vaex DataFrames or CSV files. These fingerprints help in tracking changes in data,
   and when used in combination with the caching mixins, can enhance the caching mechanisms by providing
   more unique and consistent cache keys.

Together, these components allow efficient caching of method call results, reduce processing time and
resource usage, and make it easier to identify and manage changes in data.
"""
from .cache import CacheMixin, LRUCacheMixin
from .fingerprinters import CSVFingerprinter, Fingerprinter, VaexFingerprinter


__all__ = [
    "CacheMixin",
    "LRUCacheMixin",
    "CSVFingerprinter",
    "VaexFingerprinter",
    "Fingerprinter",
]
