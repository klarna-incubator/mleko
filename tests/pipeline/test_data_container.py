"""Test suite for the `pipeline.data_container` module."""

from __future__ import annotations

from pathlib import Path

from mleko.pipeline.data_container import DataContainer


class TestDataContainer:
    """Test suite for `pipeline.data_container.DataContainer`."""

    def test_init(self):
        """Should init with data."""
        data = [Path()]
        container = DataContainer(data={"raw_data": data})
        assert container.data["raw_data"] == data

    def test_repr(self):
        """Should match string representation."""
        data = [Path()]
        container = DataContainer(data={"raw_data": data})
        assert f"{container!r}" in {
            "<DataContainer: data_type=dict, data={'raw_data': [PosixPath('.')]}>",
            "<DataContainer: data_type=list, data={'raw_data': [WindowsPath('.')]}>",
        }
