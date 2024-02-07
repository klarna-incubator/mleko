"""Test suite for the `utils.decorators` module."""

from __future__ import annotations

import inspect
from time import perf_counter, sleep
from unittest.mock import MagicMock

from mleko.utils.decorators import auto_repr, timing


class TestAutoRepr:
    """Test suite for `utils.decorators.@auto_repr`."""

    def test_no_default_args(self):
        """Should work with no default arguments."""

        class TestClass:  # noqa: B903
            @auto_repr
            def __init__(self, a, b):
                self.a = a
                self.b = b

        test_obj = TestClass(1, 2)
        assert repr(test_obj) == "TestClass(a=1, b=2)"

    def test_default_args(self):
        """Should work with default arguments."""

        class TestClass:  # noqa: B903
            @auto_repr
            def __init__(self, a=1, b=2):
                self.a = a
                self.b = b

        test_obj = TestClass()
        assert repr(test_obj) == "TestClass(a=1, b=2)"

    def test_both_args(self):
        """Should work with both default and non-default arguments."""

        class TestClass:  # noqa: B903
            @auto_repr
            def __init__(self, a, b=2):
                self.a = a
                self.b = b

        test_obj = TestClass(1)
        assert repr(test_obj) == "TestClass(a=1, b=2)"

    def test_keyword_args(self):
        """Should work with keyword arguments passed to the decorated class's __init__ method."""

        class TestClass:  # noqa: B903
            @auto_repr
            def __init__(self, a=4, b=6):
                self.a = a
                self.b = b

        test_obj = TestClass(a=1, b=2)
        assert repr(test_obj) == "TestClass(a=1, b=2)"

    def test_with_class_arguments(self):
        """Should work when argument is an object that might or might not use auto_repr decorator."""

        class TestClassWithAutoRepr:  # noqa: B903
            @auto_repr
            def __init__(self, a, b):
                self.a = a
                self.b = b

        class TestClassWithoutAutoRepr:  # noqa: B903
            def __init__(self, a, b):
                self.a = a
                self.b = b

        class MainClassWithAutoRepr:  # noqa: B903
            @auto_repr
            def __init__(self, obj):
                self.obj = obj

        class MainClassWithoutAutoRepr:  # noqa: B903
            def __init__(self, obj):
                self.obj = obj

        test_with = TestClassWithAutoRepr(1, 2)
        test_without = TestClassWithoutAutoRepr(1, 2)

        main_with_with = MainClassWithAutoRepr(test_with)
        main_with_without = MainClassWithAutoRepr(test_without)
        main_without_with = MainClassWithoutAutoRepr(test_with)
        main_without_without = MainClassWithoutAutoRepr(test_without)

        assert repr(main_with_with) == f"MainClassWithAutoRepr(obj={repr(test_with)})"
        assert repr(main_with_without) == f"MainClassWithAutoRepr(obj={test_without})"
        assert repr(main_without_with) != f"MainClassWithoutAutoRepr(obj={repr(test_with)})"
        assert repr(main_without_without) != f"MainClassWithoutAutoRepr(obj={test_without})"

    def test_preserve_signature(self):
        """Should preserve the signature of the wrapped function."""

        @auto_repr
        def sample_function(a, b, *, c=None, **kwargs):
            pass

        sig = inspect.signature(sample_function)

        assert sig == inspect.signature(sample_function.__wrapped__)


class TestTiming:
    """Test suite for `utils.decorators.@timing`."""

    class CustomLoggerMock:
        """`CustomLogger` with debug Mocked."""

        debug = MagicMock()

    def test_decorator(self):
        """Should call logger.debug."""

        @timing
        def sample_function():
            return "Test"

        sample_function.__globals__["logger"] = self.CustomLoggerMock()
        result = sample_function()
        assert result == "Test"
        assert self.CustomLoggerMock.debug.called

    def test_args_kwargs(self):
        """Should pass function with correct args and kwargs."""

        @timing
        def sample_function(a, b, c=1, d=2):
            return a + b + c + d

        sample_function.__globals__["logger"] = self.CustomLoggerMock()
        result = sample_function(3, 4, c=5, d=6)

        assert result == 18
        assert self.CustomLoggerMock.debug.called

    def test_elapsed_time(self):
        """Should log the correct elapsed time."""

        @timing
        def sample_function(sleep_time):
            sleep(sleep_time)

        sample_function.__globals__["logger"] = self.CustomLoggerMock()
        sleep_time = 0.1
        tolerance = 0.05
        ts = perf_counter()
        sample_function(sleep_time)
        te = perf_counter()
        elapsed_time = te - ts

        assert self.CustomLoggerMock.debug.called
        logged_time = float(self.CustomLoggerMock.debug.call_args[0][0].split()[-1].strip("s"))
        assert sleep_time - tolerance < logged_time < sleep_time + tolerance
        assert elapsed_time - tolerance < logged_time < elapsed_time + tolerance

    def test_preserve_signature(self):
        """Should preserve the signature of the wrapped function."""

        @timing
        def sample_function(a, b, *, c=None, **kwargs):
            pass

        sig = inspect.signature(sample_function)

        assert sig == inspect.signature(sample_function.__wrapped__)
