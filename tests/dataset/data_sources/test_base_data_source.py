"""Test suite for the `dataset.data_sources.base_data_source` module."""
from __future__ import annotations

from pathlib import Path

from mleko.dataset.data_sources.base_data_source import BaseDataSource


class TestBaseDataSource:
    """Test suite for `dataset.data_sources.base_data_source.BaseDataSource`."""

    class DataSource(BaseDataSource):
        """Test class inheriting from the `BaseDataSource`."""

        def fetch_data(self, force_recompute: bool):
            """Fetch data."""
            pass

    def test_init(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory)
        assert temporary_directory.exists()
        assert test_data._destination_directory == temporary_directory
