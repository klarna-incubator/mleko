"""Abstract base class module for data source implementations to fetch and store data from various sources."""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level custom logger."""


@dataclass
class LocalFileEntry:
    """Manifest entry for a single local file."""

    name: str
    """Name of the file."""

    size: int
    """Size of the file in bytes."""


@dataclass
class LocalManifest:
    """Manifest for the local dataset."""

    files: list[LocalFileEntry]
    """List of files in the local dataset."""


class LocalManifestHandler:
    """`LocalManifestHandler` provides a convenient interface for reading and writing local manifest files."""

    def __init__(self, manifest_path: str | Path):
        """Initializes the local manifest handler.

        The manifest is intended to be used to keep track of the downloaded file names and sizes.
        It should reflect the current state of the local dataset.

        Args:
            manifest_path: Path to the manifest file.
        """
        self._manifest_path = Path(manifest_path)

    def set_files(self, file_data: list[LocalFileEntry]) -> None:
        """Sets the manifest to the specified files.

        Args:
            file_data: List of file data to set the manifest to.
        """
        manifest_data = self._read_manifest()
        manifest_data.files = file_data
        self._write_manifest(manifest_data)

    def remove_files(self, file_names: list[str]) -> None:
        """Removes the specified files from the manifest.

        Args:
            file_names: List of file names to remove from the manifest.
        """
        manifest_data = self._read_manifest()
        manifest_data.files = [file for file in manifest_data.files if file.name not in file_names]
        self._write_manifest(manifest_data)

    def get_file_names(self) -> list[str]:
        """Gets the list of file names in the manifest.

        Returns:
            List of file names in the manifest.
        """
        manifest = self._read_manifest()
        return [file.name for file in manifest.files]

    def _read_manifest(self) -> LocalManifest:
        """Reads the manifest from the manifest file.

        Returns:
            Manifest data read from the manifest file.
        """
        try:
            if self._manifest_path.exists():
                with open(self._manifest_path) as manifest_file:
                    manifest_dict = json.load(manifest_file)
                    local_manifest = self._deserialize_manifest(manifest_dict)
                    return local_manifest
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error(f"Error decoding manifest file {self._manifest_path}: {e}")
        return LocalManifest([])

    def _write_manifest(self, manifest_data: LocalManifest) -> None:
        """Writes the manifest to the manifest file.

        Args:
            manifest_data: Manifest data to write to the manifest file.
        """
        with open(self._manifest_path, "w") as manifest_file:
            json.dump(dataclasses.asdict(manifest_data), manifest_file, indent=2)

    def _deserialize_manifest(self, manifest_dict: dict) -> LocalManifest:
        """Deserializes the manifest from the manifest dictionary.

        Args:
            manifest_dict: Manifest dictionary to deserialize.

        Returns:
            Deserialized manifest.
        """
        files = [LocalFileEntry(**file_dict) for file_dict in manifest_dict.get("files", [])]
        return LocalManifest(files=files)


class BaseIngester(ABC):
    """`BaseIngester` is an abstract base class for data source classes that interact with various external sources.

    This class provides the basic structure and methods necessary for derived data source classes, facilitating data
    fetching from various sources.
    """

    def __init__(self, cache_directory: str | Path, fingerprint: str | None) -> None:
        """Initializes the data source and ensures the destination directory exists.

        Args:
            cache_directory: Directory where the fetched data will be stored locally.
            fingerprint: Optional fingerprint to append to the destination directory.
        """
        self._cache_directory = Path(cache_directory)
        if fingerprint is not None:
            self._fingerprint = fingerprint
            self._cache_directory = self._cache_directory / fingerprint
        self._cache_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads and stores data in the 'cache_directory' using the specific data source implementation.

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
            self._cache_directory / file_name
            for file_name in file_names
            if (self._cache_directory / file_name).exists()
        ]

    def _delete_local_files(self, file_names: list[str]) -> None:
        """Deletes the specified files from the local dataset.

        Args:
            file_names: List of file names to delete from the local dataset.
        """
        for file_path in self._get_full_file_paths(file_names):
            file_path.unlink()
