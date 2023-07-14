"""Test suite for the `cache.test_lru_cache_mixin` module."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from mleko.cache.lru_cache_mixin import LRUCacheMixin


class TestLRUCacheMixin:
    """Test suite for `cache.test_lru_cache_mixin.LRUCacheMixin`."""

    class MyTestClass(LRUCacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, cache_file_suffix, max_entries):
            """Initialize cache."""
            super().__init__(cache_directory, cache_file_suffix, max_entries)

        def my_method(self, a, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a, [a], cache_group, force_recompute)[1]

    class MyTestClass2(LRUCacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, cache_file_suffix, max_entries):
            """Initialize cache."""
            super().__init__(cache_directory, cache_file_suffix, max_entries)

        def my_method(self, a, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a, [a], cache_group, force_recompute)[1]

    def test_eviction(self, temporary_directory: Path):
        """Should evict the least recently used cache entries correctly."""
        lru_cached_class = self.MyTestClass(temporary_directory, "cache", 2)
        n_calls = 3
        for i in range(n_calls):
            lru_cached_class.my_method(i)

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            lru_cached_class.my_method(0)
            patched_save_to_cache.assert_called()

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            lru_cached_class.my_method(1)
            lru_cached_class.my_method(2)
            patched_save_to_cache.assert_not_called()

    def test_force_recompute(self, temporary_directory: Path):
        """Should move back existing cached value to end if called again."""
        lru_cached_class = self.MyTestClass(temporary_directory, "cache", 2)
        n_calls = 3
        key = ""
        for i in range(n_calls):
            lru_cached_class.my_method(i)
            if i == 1:
                key = list(lru_cached_class._cache.keys())[-1]

        lru_cached_class.my_method(1, force_recompute=True)
        assert list(lru_cached_class._cache.keys())[-1] == key

    def test_clean_cache_on_load(self, temporary_directory: Path):
        """Should cleane existing cache on disk from previous object if cache is too small."""
        cache_suffix = "cache"
        cache_file_prefix_name = "d91956ef6381f61dbb4ae6b47a4fa33"
        n_cache_entries = 2
        for i in range(n_cache_entries + 3):
            new_file = temporary_directory / f"MyTestClass.test.{cache_file_prefix_name}{i}.{cache_suffix}"
            new_file.touch()
            new_file_extra = temporary_directory / f"MyTestClass.test.{cache_file_prefix_name}{i}.pkl"
            new_file_extra.touch()
            new_modified_time = datetime.timestamp(datetime.now() + timedelta(hours=i))
            os.utime(new_file, (new_modified_time, new_modified_time))

        lru_cached_class = self.MyTestClass(temporary_directory, "cache", 2)

        cache_file_keys = list(temporary_directory.glob(f"{cache_file_prefix_name}*.{cache_suffix}"))
        cache_file_endings = [int(cache_key.stem[-1]) for cache_key in cache_file_keys]
        assert len(list(temporary_directory.glob("*.pkl"))) == 2
        assert len(lru_cached_class._cache) == 1
        assert len(lru_cached_class._cache["MyTestClass.test"].keys()) == n_cache_entries
        assert all([cache_key_ending > n_cache_entries for cache_key_ending in cache_file_endings])

    def test_two_classes_same_cache(self, temporary_directory: Path):
        """Should correctly cache different classes with same arguments."""
        lru_cached_class = self.MyTestClass(temporary_directory, "cache", 2)
        lru_cached_class2 = self.MyTestClass2(temporary_directory, "cache", 2)

        lru_cached_class.my_method(1)
        lru_cached_class2.my_method(1)

        assert len(lru_cached_class._cache) == 1
        assert len(lru_cached_class2._cache) == 1
        assert len(list(temporary_directory.glob("*.cache"))) == 2

    def test_disabled_cache(self, temporary_directory: Path):
        """Should not cache if `disable_cache=True`."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", 0)

        result1 = my_test_instance.my_method(1)
        assert result1 == 1

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method(1)
            patched_save_to_cache.assert_not_called()
