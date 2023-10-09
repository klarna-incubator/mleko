"""Test suite for the `dataset.ingest.base_ingester` module."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from mleko.dataset.ingest.base_ingester import BaseIngester, LocalFileEntry, LocalManifest, LocalManifestHandler


class TestBaseIngester:
    """Test suite for `dataset.ingest.base_ingester.BaseIngester`."""

    class DataSource(BaseIngester):
        """Test class inheriting from the `BaseIngester`."""

        def fetch_data(self, _force_recompute: bool):
            """Fetch data."""
            pass

    def test_init(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory, None)
        assert temporary_directory.exists()
        assert test_data._destination_directory == temporary_directory

    def test_init_with_fingerprint(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory, "fingerprint")
        destination_directory = temporary_directory / "fingerprint"
        assert destination_directory.exists()
        assert test_data._destination_directory == destination_directory


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
