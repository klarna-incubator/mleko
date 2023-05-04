"""Test suite for the `pipeline.data_container` module."""
from __future__ import annotations

from pathlib import Path

from mleko.pipeline.data_container import DataContainer


class TestBaseDataContainer:
    """Test suite for `pipeline.data_container.BaseDataContainer`."""

    def test_init(self):
        """Should init with data."""
        data = [Path()]
        container = DataContainer(data)
        assert container.data == data

    def test_repr(self):
        """Should match string representation."""
        data = [Path()]
        container = DataContainer(data)
        assert f"{container!r}" == "<DataContainer: data_type=list, data=[PosixPath('.')]>"
