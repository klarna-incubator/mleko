"""The subpackage provides functionality for splitting `vaex` DataFrames into two separate DataFrames.

The subpackage contains the following splitter classes:
    - `BaseSplitter`: The abstract base class for all splitters.
    - `RandomSplitter`: A splitter for splitting `vaex` DataFrames into two random parts.
    - `ExpressionSplitter`: A splitter for splitting `vaex` DataFrames into two parts based on a specified expression.
"""

from __future__ import annotations

from .base_splitter import BaseSplitter
from .expression_splitter import ExpressionSplitter
from .random_splitter import RandomSplitter


__all__ = ["BaseSplitter", "RandomSplitter", "ExpressionSplitter"]
