"""Test suite for the `data.sources.base_data_class` module."""
from pathlib import Path

from mleko.data.sources.base_data_source import BaseDataSource


class TestBaseDataSource:
    """Test suite for `data.sources.base_data_class.BaseDataSource`."""

    class DataSource(BaseDataSource):
        """Test class inheriting from the `BaseDataSource`."""

        def fetch_data(self, _use_cache):
            """Fetch data."""
            pass

    def test_init(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory)
        assert temporary_directory.exists()
        assert test_data._destination_directory == temporary_directory
