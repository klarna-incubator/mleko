"""Abstract base class module for data source implementations to fetch and store data from various sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level custom logger."""


class BaseIngester(ABC):
    """`BaseIngester` is an abstract base class for data source classes that interact with various external sources.

    This class provides the basic structure and methods necessary for derived data source classes, facilitating data
    fetching from various sources.
    """

    def __init__(self, destination_directory: str | Path, fingerprint: str | None) -> None:
        """Initializes the data source and ensures the destination directory exists.

        Args:
            destination_directory: Directory where the fetched data will be stored locally.
            fingerprint: Optional fingerprint to append to the destination directory.
        """
        self._destination_directory = Path(destination_directory)
        if fingerprint is not None:
            self._fingerprint = fingerprint
            self._destination_directory = self._destination_directory / fingerprint
        self._destination_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads and stores data in the 'destination_directory' using the specific data source implementation.

        Args:
            force_recompute: Whether to force the data source to recompute its output, even if it already exists.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseIngester`.
        """
        raise NotImplementedError

    def _get_full_file_paths(self, file_names: list[str]) -> list[Path]:
        """Gets the full file paths for the specified file names.

        Note that this method only returns the file paths for files that exist locally.

        Args:
            file_names: List of file names to get the full file paths for.

        Returns:
            List of full file paths for the specified file names.
        """
        return [
            self._destination_directory / file_name
            for file_name in file_names
            if (self._destination_directory / file_name).exists()
        ]

    def _delete_local_files(self, file_names: list[str]) -> None:
        """Deletes the specified files from the local dataset.

        Args:
            file_names: List of file names to delete from the local dataset.
        """
        for file_path in self._get_full_file_paths(file_names):
            file_path.unlink()
