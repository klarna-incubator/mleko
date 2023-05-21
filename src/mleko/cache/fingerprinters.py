"""This module provides Fingerprinter classes for generating unique fingerprints.

Fingerprinters are used for generating unique fingerprints of various data and file types,
such as Vaex DataFrames or CSV files. These fingerprints can be used to track changes in data and support
caching mechanisms.
"""
from __future__ import annotations

import gzip
import hashlib
from abc import ABC, abstractmethod
from concurrent import futures
from itertools import islice
from pathlib import Path
from typing import Any

import vaex


class Fingerprinter(ABC):
    """Abstract base class for creating specialized fingerprinters."""

    @abstractmethod
    def fingerprint(self, data: Any) -> str:
        """Generate a fingerprint for the given data.

        Args:
            data: Data that should be fingerprinted.

        Raises:
            NotImplementedError: The method has to be implemented by the subclass.

        Returns:
            str: The fingerprint as a hexadecimal string.
        """
        raise NotImplementedError


class CSVFingerprinter(Fingerprinter):
    """A fingerprinter for CSV files supporting Gzipped and raw CSV files."""

    def __init__(self, n_rows: int = 1000):
        """Initialize the CSVFingerprinter.

        Args:
            n_rows: The number of rows to sample from each CSV file for fingerprinting.
        """
        self._n_rows = n_rows

    def fingerprint(self, file_paths: list[str] | list[Path]) -> str:
        """Generate a fingerprint for the given list of CSV files.

        The currently supported file types are `.csv`, `.gz`, and `.csv.gz`.

        Args:
            file_paths: A list of file paths to CSV files.

        Returns:
            The fingerprint as a hexadecimal string.
        """
        file_posix_paths: list[Path] = [Path(file_path) for file_path in file_paths]
        with futures.ThreadPoolExecutor(max_workers=None) as executor:
            file_fingerprints = list(executor.map(self._fingerprint_csv_file, file_posix_paths))

        file_fingerprints.sort()
        fingerprint = hashlib.md5("".join(file_fingerprints).encode()).hexdigest()
        return fingerprint

    def _fingerprint_csv_file(self, file_path: Path) -> str:
        """Generate a fingerprint for a single CSV file.

        Args:
            file_path: The file path to a CSV file.

        Raises:
            ValueError: File is unsupported file type.

        Returns:
            The fingerprint as a hexadecimal string.
        """
        if file_path.suffix not in {".csv", ".gz", ".csv.gz"}:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        if file_path.suffix in {".gz", ".csv.gz"}:
            with gzip.open(file_path, "rb") as f:
                sample = b"".join(islice((f.readline() for _ in range(self._n_rows)), self._n_rows))
        else:
            with open(file_path, "rb") as f:
                sample = b"".join(islice((f.readline() for _ in range(self._n_rows)), self._n_rows))
        fingerprint = hashlib.md5(str(sample).encode()).hexdigest()
        return fingerprint


class VaexFingerprinter(Fingerprinter):
    """A fingerprinter for Vaex DataFrames."""

    def fingerprint(self, dataframe: vaex.DataFrame) -> str:
        """Generate a fingerprint for a Vaex DataFrame.

        Args:
            dataframe: The Vaex DataFrame to be fingerprinted.

        Returns:
            The fingerprint as a hexadecimal string.
        """
        fingerprint = hashlib.md5(str(dataframe.fingerprint()).encode()).hexdigest()
        return fingerprint
