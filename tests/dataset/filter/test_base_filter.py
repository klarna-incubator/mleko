"""Test suite for `dataset.filter.base_filter`."""

from __future__ import annotations

from pathlib import Path

import vaex

from mleko.dataset.filter import BaseFilter


class TestBaseSplitter:
    """Test suite for `dataset.filter.base_filter.BaseFilter`."""

    class DerivedFilter(BaseFilter):
        """Test class."""

        def filter(self, _file_paths):
            """Filter."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from convert method."""
        test_derived_data_filter = self.DerivedFilter(temporary_directory, 1)

        df = test_derived_data_filter.filter([])
        assert df.shape == (3, 2)
        assert df.column_names == ["a", "b"]
