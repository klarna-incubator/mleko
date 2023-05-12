"""Test suite for the `cache.cache` module."""
from __future__ import annotations

import hashlib
import inspect
import pickle
from pathlib import Path
from typing import Hashable
from unittest.mock import patch

from mleko.cache.cache import CacheMixin, LRUCacheMixin, get_frame_qualname


class TestGetFrameQualname:
    """Test suite for `cache.cache.get_frame_qualname`."""

    def test_func(self):
        """Should correctly return fully qualified name of calling function."""

        def dummy_function():
            return get_frame_qualname(inspect.stack()[0])

        result = dummy_function()
        assert result == "tests.cache.test_cache.dummy_function"

    def test_method(self):
        """Should correctly return fully qualified name of calling method."""

        class DummyClass:
            def dummy_method(self):
                return get_frame_qualname(inspect.stack()[0])

        result = DummyClass().dummy_method()
        assert result == "tests.cache.test_cache.DummyClass.dummy_method"


class TestCacheMixin:
    """Test suite for `cache.cache.CacheMixin`."""

    class MyTestClass(CacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, cache_file_suffix):
            """Initialize cache."""
            super().__init__(cache_directory, cache_file_suffix)

        def my_method_1(self, a, b, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a + b, [a, b], force_recompute)

        def my_method_2(self, a, b, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a * b, [a, b], force_recompute)

    def test_cached_execute(self, temporary_directory: Path):
        """Should save to cache as expected."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache")

        result = my_test_instance.my_method_1(1, 2)
        assert result == 3

    def test_cache_key_computation(self, temporary_directory: Path):
        """Should compute MD5 based cache keys correctly."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache")

        dummy_frame_qualname = "module.Class.method"
        cache_keys: list[Hashable] = [1, 2]
        data = pickle.dumps((dummy_frame_qualname, cache_keys))
        expected_key = hashlib.md5(data).hexdigest()

        key = my_test_instance._compute_cache_key(cache_keys, dummy_frame_qualname)
        assert key == expected_key

    def test_different_functions_same_arguments(self, temporary_directory: Path):
        """Should correctly cache different functions with same arguments."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache")

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3, "Result of my_method_1(1, 2) should be 3"

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_2(1, 2)
            patched_save_to_cache.assert_called()

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2)
            patched_save_to_cache.assert_not_called()

    def test_forced_recompute(self, temporary_directory: Path):
        """Should recompute already cached value if `force_recompute=True`."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache")

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3, "Result of my_method_1(1, 2) should be 3"

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2, force_recompute=True)
            patched_save_to_cache.assert_called()


class TestLRUCacheMixin:
    """Test suite for `cache.cache.CacheMixin`."""

    class MyTestClass(LRUCacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, cache_file_suffix, max_entries):
            """Initialize cache."""
            super().__init__(cache_directory, cache_file_suffix, max_entries)

        def my_method(self, a, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a, [a], force_recompute)

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
        """Should clean existing cache on disk from previous object if cache is too small."""
        cache_suffix = "cache"
        cache_file_prefix_name = "d91956ef6381f61dbb4ae6b47a4fa33"
        n_cache_entries = 2
        for i in range(n_cache_entries + 3):
            (temporary_directory / f"{cache_file_prefix_name}{i}.{cache_suffix}").touch()
        lru_cached_class = self.MyTestClass(temporary_directory, "cache", 2)

        cache_file_keys = list(temporary_directory.glob(f"{cache_file_prefix_name}*.{cache_suffix}"))
        assert len(lru_cached_class._cache) == n_cache_entries
        assert len(cache_file_keys) == n_cache_entries
