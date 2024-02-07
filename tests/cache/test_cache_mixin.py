"""Test suite for the `cache.cache_mixin` module."""

from __future__ import annotations

import hashlib
import inspect
import pickle
from pathlib import Path
from typing import Hashable
from unittest.mock import patch

import joblib
import pytest

from mleko.cache.cache_mixin import CacheMixin, get_qualified_name_from_frame, get_qualified_name_of_caller
from mleko.cache.handlers.base_cache_handler import CacheHandler


class TestGetFrameQualname:
    """Test suite for `cache.cache.get_qualified_name_from_frame`."""

    def test_func(self):
        """Should correctly return fully qualified name of calling function."""

        def dummy_function():
            return get_qualified_name_from_frame(inspect.stack()[0])

        result = dummy_function()
        assert result == "tests.cache.test_cache_mixin.dummy_function"

    def test_method(self):
        """Should correctly return fully qualified name of calling method."""

        class DummyClass:
            def dummy_method(self):
                return get_qualified_name_from_frame(inspect.stack()[0])

        result = DummyClass().dummy_method()
        assert result == "tests.cache.test_cache_mixin.DummyClass.dummy_method"


class TestGetClassMethodName:
    """Test suite for `cache.cache.get_qualified_name_of_caller`."""

    def test_func(self):
        """Should correctly return fully qualified name of calling function."""

        def dummy_function():
            return get_qualified_name_of_caller(1)

        result = dummy_function()
        assert result == "test_cache_mixin.dummy_function"

    def test_method(self):
        """Should correctly return fully qualified name of calling method."""

        class DummyClass:
            def dummy_method(self):
                return get_qualified_name_of_caller(1)

        result = DummyClass().dummy_method()
        assert result == "DummyClass.dummy_method"


class TestCacheMixin:
    """Test suite for `cache.cache_mixin.CacheMixin`."""

    class MyTestClass(CacheMixin):
        """Cached test class."""

        def __init__(self, cache_directory, disable_cache):
            """Initialize cache."""
            super().__init__(cache_directory, disable_cache)

        def my_method_1(self, a, b, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a + b, [a, b], cache_group, force_recompute)

        def my_method_2(self, a, b, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: a * b, [a, b], cache_group, force_recompute)

        def my_method_3(self, list_vals, cache_group=None, force_recompute=False):
            """Cached execute."""
            return self._cached_execute(lambda: list_vals, [list_vals], cache_group, force_recompute)

        def multiple_output(self, list_vals, cache_group=None, force_recompute=False):
            """Cached execute with mulyiple outputs."""
            return self._cached_execute(
                lambda: (list_vals, list_vals),
                [list_vals],
                cache_group,
                force_recompute,
                cache_handlers=[
                    CacheHandler(
                        writer=lambda cache_file_path, output: pickle.dump(output, open(cache_file_path, "wb")),
                        reader=lambda cache_file_path: pickle.load(open(cache_file_path, "rb")),
                        suffix="pkl",
                        can_handle_none=True,
                    ),
                    CacheHandler(
                        writer=lambda cache_file_path, output: joblib.dump(output, open(cache_file_path, "wb")),
                        reader=lambda cache_file_path: joblib.load(open(cache_file_path, "rb")),
                        suffix="joblib",
                        can_handle_none=True,
                    ),
                ],
            )[1]

    def test_cached_execute(self, temporary_directory: Path):
        """Should save to cache as expected."""
        my_test_instance = self.MyTestClass(temporary_directory, False)

        result = my_test_instance.my_method_1(1, 2)
        assert result == 3

    def test_cache_key_computation(self, temporary_directory: Path):
        """Should compute MD5 based cache keys correctly."""
        my_test_instance = self.MyTestClass(temporary_directory, False)

        dummy_class_method_name = "python.pytest_pyfunc_call"
        cache_group = "cache_group"
        cache_keys: list[Hashable] = [1, 2]
        data = pickle.dumps(cache_keys)
        expected_key = f"{dummy_class_method_name}.{cache_group}.{hashlib.md5(data).hexdigest()}"

        key = my_test_instance._compute_cache_key(cache_keys, cache_group)
        assert key == expected_key

    def test_cache_key_overflow(self, temporary_directory: Path):
        """Should raise ValueError if cache key is too long."""
        my_test_instance = self.MyTestClass(temporary_directory, False)

        cache_group = "".join(["a" for _ in range(255)])
        cache_keys: list[Hashable] = [1, 2]

        with pytest.raises(ValueError):
            my_test_instance._compute_cache_key(cache_keys, cache_group)

    def test_different_functions_same_arguments(self, temporary_directory: Path):
        """Should correctly cache different functions with same arguments."""
        my_test_instance = self.MyTestClass(temporary_directory, False)

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
        my_test_instance = self.MyTestClass(temporary_directory, False)

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2, force_recompute=True)
            patched_save_to_cache.assert_called()

    def test_multiple_outputs(self, temporary_directory: Path):
        """Should successfully cache multiple outputs."""
        my_test_instance = self.MyTestClass(temporary_directory, False)
        result1 = my_test_instance.my_method_1(1, 2)
        my_test_instance = self.MyTestClass(temporary_directory, False)
        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        my_test_instance = self.MyTestClass(temporary_directory, False)
        result2 = my_test_instance.my_method_2(1, 2)
        my_test_instance = self.MyTestClass(temporary_directory, False)
        result2 = my_test_instance.my_method_2(1, 2)
        assert result2 == 2

        values_tuple = tuple(range(102))
        my_test_instance = self.MyTestClass(temporary_directory, False)
        result3 = my_test_instance.my_method_3(values_tuple)
        my_test_instance = self.MyTestClass(temporary_directory, False)
        result3 = my_test_instance.my_method_3(values_tuple)
        assert result3 == values_tuple

        assert len(list(temporary_directory.glob("*.pkl"))) == 104

    def test_disabled_cache(self, temporary_directory: Path):
        """Should not cache if `disable_cache=True`."""
        my_test_instance = self.MyTestClass(temporary_directory, True)

        result1 = my_test_instance.my_method_1(1, 2)
        assert result1 == 3

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            my_test_instance.my_method_1(1, 2)
            patched_save_to_cache.assert_not_called()

    def test_multiple_output(self, temporary_directory: Path):
        """Should successfully cache multiple outputs with different cache handlers."""
        my_test_instance = self.MyTestClass(temporary_directory, False)
        values_tuple = tuple(range(102))
        result = my_test_instance.multiple_output(values_tuple)

        assert result == values_tuple
        assert len(list(temporary_directory.glob("*.pkl"))) == 1
        assert len(list(temporary_directory.glob("*.joblib"))) == 1

        with patch.object(self.MyTestClass, "_save_to_cache") as patched_save_to_cache:
            cached_result = my_test_instance.multiple_output(values_tuple)
            patched_save_to_cache.assert_not_called()

        assert cached_result == values_tuple
