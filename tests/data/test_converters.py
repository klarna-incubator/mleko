"""Test suite for the `data.converters` module."""
from __future__ import annotations

import glob
from pathlib import Path
from unittest.mock import patch

import vaex

from mleko.data.converters import BaseDataConverter, CsvToArrowConverter
from tests.conftest import generate_csv_files


class TestBaseDataConverter:
    """Test suite for `data.converters.BaseDataConverter`."""

    class DerivedDataConverter(BaseDataConverter):
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


class TestCsvToArrowConverter:
    """Test suite for `data.converters.CsvToArrowConverter`."""

    def test_convert(self, temporary_directory: Path):
        """Should convert CSV files to arrow files using '_convert' and save them to the output directory."""
        csv_to_arrow_converter = CsvToArrowConverter(temporary_directory)

        n_files = 2
        file_paths = generate_csv_files(temporary_directory, n_files)
        csv_to_arrow_converter._convert(file_paths)

        arrow_files = list(temporary_directory.glob("df_chunk_*.arrow"))
        assert len(arrow_files) == n_files

        dfs = [vaex.open(f) for f in arrow_files]

        for df in dfs:
            assert str(list(df.dtypes)) == "[datetime64[s], float64, string, bool]"
            assert df.column_names == ["Time", "Count", "Name", "Is Best"]
            assert df.shape == (3, 4)
            assert df.Name.countna() == 1
            df.close()

    def test_cache_hit(self, temporary_directory: Path):
        """Should convert a number of CSV files to arrow and return cached values on second call."""
        csv_to_arrow_converter = CsvToArrowConverter(
            temporary_directory,
            forced_numerical_columns=["Count"],
            forced_categorical_columns=["Name"],
            forced_boolean_columns=["Is Best"],
            num_workers=1,
        )

        n_files = 1
        file_paths = generate_csv_files(temporary_directory, n_files)
        df = csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        assert str(list(df.dtypes)) == "[datetime64[s], float64, string, bool]"
        assert df.column_names == ["Time", "Count", "Name", "Is Best"]
        assert df.shape == (n_files * 3, 4)
        assert df.Name.countna() == n_files
        assert len(glob.glob(str(temporary_directory / "df_chunk_*.arrow"))) == 0
        assert len(glob.glob(str(temporary_directory / "*.arrow"))) == 1

        with patch.object(CsvToArrowConverter, "_convert") as patched_convert:
            new_csv_to_arrow_converter = CsvToArrowConverter(
                temporary_directory,
                forced_numerical_columns=["Count"],
                forced_categorical_columns=["Name"],
                forced_boolean_columns=["Is Best"],
                num_workers=1,
            )
            new_csv_to_arrow_converter.convert(file_paths, force_recompute=False)
            patched_convert.assert_not_called()
        df.close()

    def test_cache_miss(self, temporary_directory: Path):
        """Should convert a number of CSV files to arrow and cache miss on second call."""
        csv_to_arrow_converter = CsvToArrowConverter(
            temporary_directory, downcast_float=True, num_workers=1, max_cache_entries=2
        )

        n_files = 1
        file_paths = generate_csv_files(temporary_directory, n_files, gzipped=True)
        df = csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        new_csv_to_arrow_converter = CsvToArrowConverter(
            temporary_directory, downcast_float=False, num_workers=1, max_cache_entries=2
        )
        df_new = new_csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        assert len(glob.glob(str(temporary_directory / "*.arrow"))) == 2
        df.close()
        df_new.close()
