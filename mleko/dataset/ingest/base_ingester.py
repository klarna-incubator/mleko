"""Abstract base class module for data source implementations to fetch and store data from various sources."""
from __future__ import annotations

import glob
from abc import ABC, abstractmethod
from pathlib import Path


class BaseIngester(ABC):
    """`BaseIngester` is an abstract base class for data source classes that interact with various external sources.

    This class provides the basic structure and methods necessary for derived data source classes, facilitating data
    fetching from various sources.
    """

    def __init__(self, destination_directory: str | Path) -> None:
        """Initializes the data source and ensures the destination directory exists.

        Args:
            destination_directory: Directory where the fetched data will be stored locally.
        """
        self._destination_directory = Path(destination_directory)
        self._destination_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads and stores data in the 'destination_directory' using the specific data source implementation.

        Args:
            force_recompute: Whether to force the data source to recompute its output, even if it already exists.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseIngester`.

        Returns:
            A list of Path objects pointing to the downloaded data files.
        """
        raise NotImplementedError

    def _get_local_filenames(self, file_suffixes: list[str]) -> list[Path]:
        """Retrieves local filenames for files with specified suffixes in the destination directory.

        Args:
            file_suffixes: List of file suffixes that should be returned from the destination directory,
                such as `gz`, `zip`, and `csv`.

        Returns:
            A list of Path objects for all CSV, ZIP and GZ files in the destination directory.
        """
        return sorted(
            [
                Path(filepath)
                for filepath in [
                    p for suffix in file_suffixes for p in glob.glob(f"{self._destination_directory}/*.{suffix}")
                ]
            ]
        )
