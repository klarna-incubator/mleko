"""This module provides utility functions for file and directory operations."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from .custom_logger import CustomLogger


def clear_directory(directory: Path, pattern: str = "*") -> None:
    """Remove all files in a directory that match a given pattern.

    This function takes a directory and, using the provided pattern, searches for all matching files
    and removes them. This is useful when cleaning up temporary or intermediate files in a workspace.

    Args:
        directory: The `Path` object referring to the directory to be cleared.
        pattern: The search pattern to match the files in the directory (default: "*", matches all files).
    """
    for f in directory.glob(pattern):
        f.unlink()


logger = CustomLogger()
"""A module-level custom logger."""


@dataclass
class LocalFileEntry:
    """Manifest entry for a single local file."""

    name: str
    """Name of the file."""

    size: int
    """Size of the file in bytes."""

    hash: str | None = None


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

    def add_files(self, file_data: list[LocalFileEntry]) -> None:
        """Adds the specified files to the manifest.

        Args:
            file_data: List of file data to add to the manifest.
        """
        manifest_data = self._read_manifest()
        manifest_data.files.extend(file_data)
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
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
