"""Test suite for `dataset.convert.csv_to_vaex_converter`."""

from __future__ import annotations

import glob
from pathlib import Path
from unittest.mock import patch

import vaex

from mleko.dataset.convert.csv_to_vaex_converter import CSVToVaexConverter
from tests.conftest import generate_csv_files


class TestCSVToVaexConverter:
    """Test suite for `dataset.convert.csv_to_vaex_converter.CSVToVaexConverter`."""

    def test_convert(self, temporary_directory: Path):
        """Should convert CSV files to arrow files using '_convert' and save them to the output directory."""
        csv_to_arrow_converter = CSVToVaexConverter(temporary_directory)

        n_files = 2
        file_paths = generate_csv_files(temporary_directory, n_files)
        csv_to_arrow_converter._convert(file_paths)

        arrow_files = list(temporary_directory.glob("df_chunk_*.arrow"))
        assert len(arrow_files) == n_files

        dfs = [vaex.open(f) for f in arrow_files]

        for df in dfs:
            assert str(list(df.dtypes)) == "[datetime64[s], datetime64[s], float64, string, bool, null, string]"
            assert df.column_names == ["Time", "Date", "Count", "Name", "Is_Best", "Extra_Column", "class"]
            assert df.shape == (3, 7)
            assert df.Name.countna() == 1
            df.close()

    def test_cache_hit(self, temporary_directory: Path):
        """Should convert a number of CSV files to arrow and return cached values on second call."""
        csv_to_arrow_converter = CSVToVaexConverter(
            temporary_directory,
            forced_numerical_columns=["Count"],
            forced_categorical_columns=["Name"],
            forced_boolean_columns=["Is_Best"],
            meta_columns=["Extra_Column"],
            drop_rows_with_na_columns=["Name"],
            num_workers=1,
        )

        n_files = 1
        file_paths = generate_csv_files(temporary_directory, n_files)
        _, df = csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        assert str(list(df.dtypes)) == "[datetime64[s], datetime64[s], float64, string, string, string, string]"
        assert df.column_names == ["Time", "Date", "Count", "Name", "Is_Best", "Extra_Column", "_class"]
        assert df.shape == (n_files * 2, 7)
        assert df.Name.countna() == 0
        assert len(glob.glob(str(temporary_directory / "df_chunk_*.arrow"))) == 0
        assert len(glob.glob(str(temporary_directory / "*.hdf5"))) == 1

        with patch.object(CSVToVaexConverter, "_convert") as patched_convert:
            new_csv_to_arrow_converter = CSVToVaexConverter(
                temporary_directory,
                forced_numerical_columns=["Count"],
                forced_categorical_columns=["Name"],
                forced_boolean_columns=["Is_Best"],
                meta_columns=["Extra_Column"],
                drop_rows_with_na_columns=["Name"],
                num_workers=1,
            )
            new_csv_to_arrow_converter.convert(file_paths, force_recompute=False)
            patched_convert.assert_not_called()
        df.close()

    def test_cache_miss(self, temporary_directory: Path):
        """Should convert a number of CSV files to arrow and cache miss on second call."""
        csv_to_arrow_converter = CSVToVaexConverter(
            temporary_directory, downcast_float=True, num_workers=1, cache_size=2
        )

        n_files = 1
        file_paths = generate_csv_files(temporary_directory, n_files, gzipped=True)
        _, df = csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        new_csv_to_arrow_converter = CSVToVaexConverter(
            temporary_directory, downcast_float=False, num_workers=1, cache_size=2
        )
        _, df_new = new_csv_to_arrow_converter.convert(file_paths, force_recompute=False)

        assert len(glob.glob(str(temporary_directory / "*.hdf5"))) == 2
        df.close()
        df_new.close()
