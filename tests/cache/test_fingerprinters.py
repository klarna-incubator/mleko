"""Test suite for the `cache.fingerprinters` module."""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.cache import CSVFingerprinter, VaexFingerprinter
from tests.conftest import generate_csv_files


class TestCSVFingerprinter:
    """Test suite for `cache.fingerprinters.CSVFingerprinter`."""

    def test_stable_output(self, temporary_directory: Path):
        """Should produce the same output over multiple runs, even with different file names.

        Also tests that the fingerprinter unpacks Gzipped CSV before computing the fingerprint and that neither
        order of files or mix of Gzip and raw matters for the fingerprint.
        """
        file_paths = generate_csv_files(temporary_directory, 5, gzipped=True)
        csv_fingerprinter = CSVFingerprinter(1000)
        original_fingerprint = csv_fingerprinter.fingerprint(file_paths)

        file_paths = generate_csv_files(temporary_directory, 5)
        new_csv_fingerprinter = CSVFingerprinter(250)
        new_fingerprint = new_csv_fingerprinter.fingerprint(file_paths)

        assert original_fingerprint == new_fingerprint

    def test_on_different_n_rows(self, temporary_directory: Path):
        """Should produce different values when the number of rows read are not equal."""
        file_paths = generate_csv_files(temporary_directory, 5, gzipped=True)
        csv_fingerprinter = CSVFingerprinter(4)
        original_fingerprint = csv_fingerprinter.fingerprint(file_paths)

        new_csv_fingerprinter = CSVFingerprinter(3)
        new_fingerprint = new_csv_fingerprinter.fingerprint(file_paths)

        assert original_fingerprint != new_fingerprint

    def test_detects_single_change(self, temporary_directory: Path):
        """Should produce the same output over multiple runs."""
        file_paths = generate_csv_files(temporary_directory, 5)

        csv_fingerprinter = CSVFingerprinter()
        original_fingerprint = csv_fingerprinter.fingerprint(file_paths)

        with open(file_paths[0], "a") as f:
            f.write("s")
        new_csv_fingerprinter = CSVFingerprinter()
        new_fingerprint = new_csv_fingerprinter.fingerprint(file_paths)

        assert original_fingerprint != new_fingerprint

    def test_unsupported_file_type(self):
        """Should throw a ValueError if an unsupported file type is supplied."""
        file_paths = [Path("file.arrow")]

        csv_fingerprinter = CSVFingerprinter()
        value_error = False
        try:
            csv_fingerprinter.fingerprint(file_paths)
        except ValueError:
            value_error = True
        assert value_error is True


class TestVaexFingerprinter:
    """Test suite for `cache.fingerprinters.VaexFingerprinter`."""

    def test_stable_output(self):
        """Should produce the same output over multiple runs even when."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(df)
        new_fingerprint = VaexFingerprinter().fingerprint(df)
        assert original_fingerprint == new_fingerprint

    def test_on_different_n_rows(self, temporary_directory: Path):
        """Should produce same fingerprint on two dataframes with same content."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        original_df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(original_df)

        new_df = vaex.from_dict(data)
        new_fingerprint = VaexFingerprinter().fingerprint(new_df)
        assert original_fingerprint == new_fingerprint

    def test_detects_single_change(self):
        """Should have different fingerprint on small change to DataFrame."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        original_df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(original_df)

        data["column1"][0] = 2
        new_df = vaex.from_dict(data)
        new_fingerprint = VaexFingerprinter().fingerprint(new_df)
        assert original_fingerprint != new_fingerprint
