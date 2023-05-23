"""This module contains the `BaseConverter` class, which provides an interface for converting file formats.

Subclasses of `BaseConverter` should implement the `convert` method, which takes a list of file paths and returns a
`vaex.DataFrame` object.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex


class BaseConverter(ABC):
    """A base class for data converter classes, providing an interface for converting file formats."""

    def __init__(self, output_directory: str | Path):
        """Initialize the BaseConverter with the output directory for the converted files.

        Args:
            output_directory: The directory where the converted files will be saved.
        """
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def convert(self, file_paths: list[Path] | list[str], force_recompute: bool = False) -> vaex.DataFrame:
        """Abstract method to convert the input file paths to the desired output format.

        Args:
            file_paths: A list of input file paths to be converted.
            force_recompute: If set to True, forces recomputation and ignores the cache.

        Returns:
            vaex.DataFrame: The resulting DataFrame after conversion.
        """
        raise NotImplementedError
