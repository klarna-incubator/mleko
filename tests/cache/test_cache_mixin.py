"""Test suite for the `cache.cache_mixin` module."""
from __future__ import annotations

import hashlib
import inspect
import pickle
from pathlib import Path
from typing import Hashable
from unittest.mock import patch

import pytest

from mleko.cache.cache_mixin import CacheMixin, get_class_method_name, get_frame_qualname


class TestGetFrameQualname:
    """Test suite for `cache.cache.get_frame_qualname`."""

    def test_func(self):
        """Should correctly return fully qualified name of calling function."""

        def dummy_function():
            return get_frame_qualname(inspect.stack()[0])

        result = dummy_function()
        assert result == "tests.cache.test_cache_mixin.dummy_function"

    def test_method(self):
        """Should correctly return fully qualified name of calling method."""

        class DummyClass:
            def dummy_method(self):
                return get_frame_qualname(inspect.stack()[0])

        result = DummyClass().dummy_method()
        assert result == "tests.cache.test_cache_mixin.DummyClass.dummy_method"


class TestGetClassMethodName:
    """Test suite for `cache.cache.get_class_method_name`."""

    def test_func(self):
        """Should correctly return fully qualified name of calling function."""

        def dummy_function():
            return get_class_method_name(1)

        result = dummy_function()
        assert result == "test_cache_mixin.dummy_function"

    def test_method(self):
        """Should correctly return fully qualified name of calling method."""

        class DummyClass:
            def dummy_method(self):
                return get_class_method_name(1)

        result = DummyClass().dummy_method()
        assert result == "DummyClass.dummy_method"


class TestCacheMixin:
    """Test suite for `cache.cache_mixin.CacheMixin`."""

    class MyTestClass(CacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, cache_file_suffix, disable_cache):
            """Initialize cache."""
            super().__init__(cache_directory, cache_file_suffix, disable_cache)

        def my_method_1(self, a, b, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a + b, [a, b], cache_group, force_recompute)[1]

        def my_method_2(self, a, b, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a * b, [a, b], cache_group, force_recompute)[1]

        def my_method_3(self, list_vals, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: list_vals, [list_vals], cache_group, force_recompute)[1]

    def test_cached_execute(self, temporary_directory: Path):
        """Should save to cache as expected."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)

        result = my_test_instance.my_method_1(1, 2)
        assert result == 3

    def test_cache_key_computation(self, temporary_directory: Path):
        """Should compute MD5 based cache keys correctly."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)

        dummy_class_method_name = "python.pytest_pyfunc_call"
        cache_group = "cache_group"
        cache_keys: list[Hashable] = [1, 2]
        data = pickle.dumps(cache_keys)
        expected_key = f"{dummy_class_method_name}.{cache_group}.{hashlib.md5(data).hexdigest()}"

        key = my_test_instance._compute_cache_key(cache_keys, cache_group)
        assert key == expected_key

    def test_cache_key_overflow(self, temporary_directory: Path):
        """Should raise ValueError if cache key is too long."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)

        cache_group = "".join(["a" for _ in range(255)])
        cache_keys: list[Hashable] = [1, 2]

        with pytest.raises(ValueError):
            my_test_instance._compute_cache_key(cache_keys, cache_group)

    def test_different_functions_same_arguments(self, temporary_directory: Path):
        """Should correctly cache different functions with same arguments."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_2(1, 2)
            patched_save_to_cache.assert_called()

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2)
            patched_save_to_cache.assert_not_called()

    def test_forced_recompute(self, temporary_directory: Path):
        """Should recompute already cached value if `force_recompute=True`."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2, force_recompute=True)
            patched_save_to_cache.assert_called()

    def test_multiple_outputs(self, temporary_directory: Path):
        """Should successfully cache multiple outputs."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result1 = my_test_instance.my_method_1(1, 2)
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result2 = my_test_instance.my_method_2(1, 2)
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result2 = my_test_instance.my_method_2(1, 2)
        assert result2 == 2

        values_tuple = tuple(range(102))
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result3 = my_test_instance.my_method_3(values_tuple)
        my_test_instance = self.MyTestClass(temporary_directory, "cache", False)
        result3 = my_test_instance.my_method_3(values_tuple)
        assert result3 == values_tuple

        assert len(list(temporary_directory.glob("*.cache"))) == 104

    def test_disabled_cache(self, temporary_directory: Path):
        """Should not cache if `disable_cache=True`."""
        my_test_instance = self.MyTestClass(temporary_directory, "cache", True)

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2)
            patched_save_to_cache.assert_not_called()
