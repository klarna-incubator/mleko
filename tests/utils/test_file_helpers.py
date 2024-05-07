"""Test suite for the `utils.file_helpers` module."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from mleko.utils.file_helpers import LocalFileEntry, LocalManifest, LocalManifestHandler, clear_directory


class TestClearDirectory:
    """Test suite for `utils.file_helpers.clear_directory`."""

    def test_clear_directory(self, temporary_directory: Path):
        """Should delete all files."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()

        clear_directory(temporary_directory)
        remaining_files = list(temporary_directory.iterdir())
        assert len(remaining_files) == 0

    def test_with_pattern(self, temporary_directory: Path):
        """Should only delete pattern matched files."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()
            (temporary_directory / f"file{i}.log").touch()

        clear_directory(temporary_directory, "*.log")
        remaining_files_log = list(temporary_directory.glob("*.log"))
        assert len(remaining_files_log) == 0

        remaining_files_txt = list(temporary_directory.glob("*.txt"))
        assert len(remaining_files_txt) == 3

    def test_no_files(self, temporary_directory: Path):
        """Should handle empty directory."""
        assert len(list(temporary_directory.iterdir())) == 0

        clear_directory(temporary_directory)

    def test_no_matching_pattern(self, temporary_directory: Path):
        """Should handle no files matching pattern."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()

        assert len(list(temporary_directory.glob("*.txt"))) == 3

        clear_directory(temporary_directory, "*.log")


class TestLocalManifestHandler:
    """Test suite for `dataset.ingest.base_ingester.LocalManifestHandler`."""

    def test_set_files(self, temporary_directory: Path):
        """Should correctly set files in the manifest."""
        manifest_handler = LocalManifestHandler(temporary_directory / "manifest.json")
        file_entries = [LocalFileEntry("file1.txt", 100), LocalFileEntry("file2.txt", 200)]
        manifest_handler.set_files(file_entries)

        with open(temporary_directory / "manifest.json") as file:
            data = json.load(file)
        assert data == dataclasses.asdict(LocalManifest(file_entries))

    def test_set_malformed(self, temporary_directory: Path):
        """Should return empty list if error."""
        with open(temporary_directory / "manifest.json", "w") as f:
            f.write('{ "key": "value"')

        manifest_handler = LocalManifestHandler(temporary_directory / "manifest.json")
        assert manifest_handler.get_file_names() == []

    def test_remove_files(self, temporary_directory: Path):
        """Should correctly remove files from the manifest."""
        manifest_handler = LocalManifestHandler(temporary_directory / "manifest.json")
        file_entries = [LocalFileEntry("file1.txt", 100), LocalFileEntry("file2.txt", 200)]
        manifest_handler.set_files(file_entries)

        manifest_handler.remove_files(["file1.txt"])

        with open(temporary_directory / "manifest.json") as file:
            data = json.load(file)
        assert data == dataclasses.asdict(LocalManifest([LocalFileEntry("file2.txt", 200)]))

    def test_get_file_names(self, temporary_directory: Path):
        """Should correctly get file names from the manifest."""
        manifest_handler = LocalManifestHandler(temporary_directory / "manifest.json")
        file_entries = [LocalFileEntry("file1.txt", 100), LocalFileEntry("file2.txt", 200)]
        manifest_handler.set_files(file_entries)

        file_names = manifest_handler.get_file_names()

        assert file_names == ["file1.txt", "file2.txt"]
