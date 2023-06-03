"""Test suite for the `dataset.ingest.base_ingester` module."""
from __future__ import annotations

from pathlib import Path

from mleko.dataset.ingest.base_ingester import BaseIngester


class TestBaseIngester:
    """Test suite for `dataset.ingest.base_ingester.BaseIngester`."""

    class DataSource(BaseIngester):
        """Test class inheriting from the `BaseIngester`."""

        def fetch_data(self, force_recompute: bool):
            """Fetch data."""
            pass

    def test_init(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory)
        assert temporary_directory.exists()
        assert test_data._destination_directory == temporary_directory
