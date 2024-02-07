"""The subpackage provides common cache handlers for serializing and deserializing data to the disk.

To implement a custom cache handler, you need to create a `CacheHandler` object with the following attributes:
    - `writer`: A function that takes a Path object and data as input and saves the data to the cache file.
    - `reader`: A function that takes a Path object as input and returns the deserialized data stored in the cache file.
    - `suffix`: The suffix of the cache files.

The following cache handlers are provided by the subpackage:
    - `JOBLIB_CACHE_HANDLER`: A cache handler for Python objects using joblib
    - `PICKLE_CACHE_HANDLER`: A cache handler for pickling Python objects.
    - `VAEX_DATAFRAME_CACHE_HANDLER`: A cache handler for `vaex` DataFrames.
"""

from .base_cache_handler import CacheHandler
from .joblib_cache_handler import JOBLIB_CACHE_HANDLER, read_joblib, write_joblib
from .pickle_cache_handler import PICKLE_CACHE_HANDLER, read_pickle, write_pickle
from .vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER, read_vaex_dataframe, write_vaex_dataframe


__all__ = [
    "CacheHandler",
    "JOBLIB_CACHE_HANDLER",
    "PICKLE_CACHE_HANDLER",
    "VAEX_DATAFRAME_CACHE_HANDLER",
    "read_joblib",
    "write_joblib",
    "read_pickle",
    "write_pickle",
    "read_vaex_dataframe",
    "write_vaex_dataframe",
]
