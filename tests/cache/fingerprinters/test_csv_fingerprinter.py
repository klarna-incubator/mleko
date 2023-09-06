"""Test suite for the `cache.fingerprinters.csv_fingerprinter`."""
from __future__ import annotations

from pathlib import Path

import pytest

from mleko.cache.fingerprinters.csv_fingerprinter import CSVFingerprinter
from tests.conftest import generate_csv_files


class TestCSVFingerprinter:
    """Test suite for `cache.fingerprinters.csv_fingerprinter.CSVFingerprinter`."""

    @pytest.fixture(autouse=True)
    def setup(self, temporary_directory: Path):
        """Generate csv files for testing."""
        self.file_paths = generate_csv_files(temporary_directory, 5)
        self.csv_fingerprinter = CSVFingerprinter()

    def test_stable_output(self):
        """Should produce the same output over multiple runs, even with different file names."""
        original_fingerprint = self.csv_fingerprinter.fingerprint(self.file_paths)
        new_csv_fingerprinter = CSVFingerprinter(250)
        new_fingerprint = new_csv_fingerprinter.fingerprint(self.file_paths)

        assert original_fingerprint == new_fingerprint

    def test_on_different_n_rows(self):
        """Should produce different values when the number of rows read are not equal."""
        original_fingerprint = self.csv_fingerprinter.fingerprint(self.file_paths)
        new_csv_fingerprinter = CSVFingerprinter(3)
        new_fingerprint = new_csv_fingerprinter.fingerprint(self.file_paths)

        assert original_fingerprint != new_fingerprint

    def test_detects_single_change(self):
        """Should produce the same output over multiple runs."""
        original_fingerprint = self.csv_fingerprinter.fingerprint(self.file_paths)

        with open(self.file_paths[0], "a") as f:
            f.write("s")
        new_fingerprint = self.csv_fingerprinter.fingerprint(self.file_paths)

        assert original_fingerprint != new_fingerprint

    def test_unsupported_file_type(self):
        """Should throw a ValueError if an unsupported file type is supplied."""
        file_paths = [Path("file.arrow")]
        value_error = False
        try:
            self.csv_fingerprinter.fingerprint(file_paths)
        except ValueError:
            value_error = True
        assert value_error is True
