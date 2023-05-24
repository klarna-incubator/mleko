"""Test suite for `dataset.convert.base_convert`."""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.dataset.convert.base_converter import BaseConverter


class TestBaseConverter:
    """Test suite for `dataset.convert.base_converter.BaseConverter`."""

    class DerivedDataConverter(BaseConverter):
        """Test class."""

        def convert(self, _file_paths):
            """Convert."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from convert method."""
        test_derived_data_converter = self.DerivedDataConverter(temporary_directory)

        df = test_derived_data_converter.convert([])
        assert df.shape == (3, 2)
        assert df.column_names == ["a", "b"]
