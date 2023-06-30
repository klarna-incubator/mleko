"""PyTest session fixture defintions and utils."""
from __future__ import annotations

import csv
import gzip
import tempfile
import uuid
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temporary_directory() -> Generator[Path, None, None]:
    """Create a temporary directory using the `tempfile` module and provide its path as a `Path` object.

    Yields:
        Path: A `Path` object representing the path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        yield Path(temporary_directory)


def generate_csv_files(directory_path: Path, n_files: int, gzipped: bool = False) -> list[Path]:
    """Generate a number CSV sample files to the specified directory.

    Each CSV sample file has one header row and three data rows with four columns each.
    The files names are UUIDs meaning file names are completely random and will never repeat.

    Args:
        directory_path: Path to write the CSV to.
        n_files: Number of files to generate.
        gzipped: Should Gzip the CSV files.

    Returns:
        List of Paths to CSV files.
    """
    file_paths: list[Path] = []
    for _ in range(n_files):
        file_path = directory_path / f"{uuid.uuid4()}.csv"
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Date", "Count", "Name", "Is Best"])
            writer.writerow(["2023-01-01 20:00:00", "2023-01-01", 3, "Linux", False])
            writer.writerow(["2023-01-01 20:00:00", "2023-01-01", 5.4, "Windows", False])
            writer.writerow(["2023-01-01 20:00:00", "2023-01-01", -1, "-9999", True])

        if gzipped:
            with open(file_path, "rb") as f_in, gzip.open(file_path.with_suffix(".gz"), "wb") as f_out:
                f_out.writelines(f_in)
            file_path.unlink()
            file_path = file_path.with_suffix(".gz")

        file_paths.append(file_path)

    return file_paths
