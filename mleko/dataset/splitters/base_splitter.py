"""Abstract base class for data splitters.

This module provides a common interface for splitting a dataframe into two parts by implementing the `split` method.
Classes that inherit from `BaseSplitter` must implement the `split` method. The `split` method should split the given
dataframe into two parts and return them as a tuple.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex


class BaseSplitter(ABC):
    """Abstract base class for data splitters.

    Provides a common interface for splitting a dataframe into two parts by implementing the `split` method.
    """

    def __init__(self, output_directory: str | Path):
        """Initializes the `BaseSplitter` with an output directory.

        Args:
            output_directory: The target directory where the split dataframes are to be saved.
        """
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def split(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        The implementation of this method should split the given dataframe into two parts and return them as a tuple.

        Args:
            dataframe: The dataframe to be split.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.

        Returns:
            A tuple containing the split dataframes.
        """
        raise NotImplementedError
