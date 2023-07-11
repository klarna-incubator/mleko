"""Test suite for the `cache.format.vaex_arrow_cache_format_mixin` module.""" ""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.cache.format.vaex_cache_format_mixin import VaexCacheFormatMixin
from mleko.cache.lru_cache_mixin import LRUCacheMixin


class TestVaexCacheFormatMixin:
    """Test suite for `cache.format.vaex_arrow_cache_format_mixin.VaexCacheFormatMixin`."""

    class MyTestClass(VaexCacheFormatMixin, LRUCacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, max_entries):
            """Initialize cache."""
            LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, max_entries, False)

        def my_method(self, a, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a, [a.fingerprint()], None, force_recompute)[1]

    def test_vaex_dataframe_arrow_mixin(self, temporary_directory: Path):
        """Should save to cache as expected."""
        my_test_instance = self.MyTestClass(temporary_directory, 2)

        df = my_test_instance.my_method(vaex.from_arrays(x=[1, 2, 3]))
        assert df.x.tolist() == [1, 2, 3]

        df = my_test_instance.my_method(vaex.from_arrays(x=[1, 2, 3]))
        assert df.x.tolist() == [1, 2, 3]

        assert len(list(temporary_directory.glob("*.hdf5"))) == 1
