"""Test suite for `dataset.split.base_splitter`."""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.dataset.split.base_splitter import BaseSplitter


class TestBaseSplitter:
    """Test suite for `dataset.split.base_splitter.BaseSplitter`."""

    class DerivedSplitter(BaseSplitter):
        """Test class."""

        def split(self, _file_paths):
            """Split."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6]), vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from convert method."""
        test_derived_data_splitter = self.DerivedSplitter(temporary_directory)

        df_train, df_test = test_derived_data_splitter.split([])
        assert df_train.shape == (3, 2)
        assert df_train.column_names == ["a", "b"]
        assert df_test.shape == (3, 2)
        assert df_test.column_names == ["a", "b"]
