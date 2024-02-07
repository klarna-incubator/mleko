"""This module provides utility decorators for classes and functions."""

from __future__ import annotations

import inspect
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar, cast

from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A CustomLogger instance that's used throughout the module for logging."""

F = TypeVar("F", bound=Callable[..., Any])
"""A TypeVar used as a generic function type throughout the module.

This TypeVar is designed for type hinting in decorators in the module. It essentially states that F can be
a function with any number of arguments and any return type.
"""


def auto_repr(init_method: F) -> F:
    """Decorator for generating a `__repr__` method for a class automatically based on the `__init__` method signature.

    The decorator inspects the `__init__` method's signature and uses parameter names and values to create a `__repr__`
    method that represents the class instance.

    Args:
        init_method: The `__init__` method of the class to be decorated.

    Returns:
        The wrapped `__init__` method with an automatically generated `__repr__` method.

    Examples:
        >>> from mleko.utils import auto_repr
        >>> class TestClass:
        ...     @auto_repr
        ...     def __init__(self, param1: int, param2: str = "default"):
        ...         self.param1 = param1
        ...         self.param2 = param2
        >>> test_instance = TestClass(1, "test")
        >>> test_instance
        TestClass(param1=1, param2='test')
    """

    @wraps(init_method)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init_method(self, *args, **kwargs)

        signature = inspect.signature(init_method)
        parameter_names = list(signature.parameters.keys())[1:]
        parameter_values = dict(zip(parameter_names, args))

        for key, param in signature.parameters.items():
            if param.default != inspect.Parameter.empty:
                parameter_values.setdefault(key, param.default)

        parameter_values.update(kwargs)

        self._auto_repr_args_kwargs = parameter_values
        self.__class__.__repr__ = generated_repr

    def generated_repr(self: Any) -> str:
        cls_name = type(self).__name__
        params_str = ", ".join(f"{key}={repr(value)}" for key, value in self._auto_repr_args_kwargs.items())
        return f"{cls_name}({params_str})"

    return cast(F, wrapped_init)


def timing(func: F) -> F:
    """A decorator that logs the execution time of the decorated function using a CustomLogger instance.

    Warning:
        On certain platforms, the timing may be inaccurate due to OS scheduling. This is a known issue with some
        MacOS versions.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that logs its execution time.

    Example:
        >>> from mleko.utils import timing
        >>> @timing
        ... def test_func():
        ...     pass
        >>> test_func()
        Function: test_func    Timing: 0.0000s
    """

    @wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> Any:
        ts = perf_counter()
        result = func(*args, **kwargs)
        te = perf_counter()
        elapsed_time = te - ts
        logger.debug(f"Function: {func.__qualname__}    Timing: {elapsed_time:.4f}s")
        return result

    return cast(F, wrap)
