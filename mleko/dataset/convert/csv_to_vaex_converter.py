"""The module contains the `CSVToVaexConverter`, which converts CSV to a random-access `vaex` compatible format."""
from __future__ import annotations

import multiprocessing
from concurrent import futures
from itertools import repeat
from pathlib import Path

import vaex
from pyarrow import csv as arrow_csv
from tqdm import tqdm

from mleko.cache.fingerprinters import CSVFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.file_helpers import clear_directory

from .base_converter import BaseConverter


logger = CustomLogger()
"""A module-level logger instance."""

V_CPU_COUNT = multiprocessing.cpu_count()
"""A module-level constant representing the total number of CPUs available on the current system."""


class CSVToVaexConverter(BaseConverter):
    """A class that converts CSV to a random-access `vaex` compatible format."""

    @auto_repr
    def __init__(
        self,
        output_directory: str | Path,
        forced_numerical_columns: list[str] | tuple[str, ...] | tuple[()] = (),
        forced_categorical_columns: list[str] | tuple[str, ...] | tuple[()] = (),
        forced_boolean_columns: list[str] | tuple[str, ...] | tuple[()] = (),
        drop_columns: list[str] | tuple[str, ...] | tuple[()] = (),
        na_values: list[str]
        | tuple[str, ...]
        | tuple[()] = (
            "-9998",
            "-9998.0",
            "-9999",
            "-9999.0",
            "-99",
            "-99.0",
            "nan",
            "none",
            "non",
            "Nan",
            "None",
            "Non",
            "",
            "N/A",
            "N/a",
            "unknown",
            "missing",
        ),
        true_values: list[str] | tuple[str, ...] | tuple[()] = ("t", "True", "true", "1"),
        false_values: list[str] | tuple[str, ...] | tuple[()] = ("f", "False", "false", "0"),
        downcast_float: bool = False,
        random_state: int | None = None,
        num_workers: int = V_CPU_COUNT,
        cache_size: int = 1,
    ) -> None:
        """Initializes the `CSVToArrowConverter` with the necessary configurations and parameters.

        Args:
            output_directory: The directory where the converted files will be saved.
            forced_numerical_columns: A sequence of column names to force as numerical type.
            forced_categorical_columns: A sequence of column names to force as categorical type.
            forced_boolean_columns: A sequence of column names to force as boolean type.
            drop_columns: A sequence of column names to drop during conversion.
            na_values: A sequence of strings to consider as NaN or missing values.
            true_values: A sequence of strings to consider as True values.
            false_values: A sequence of strings to consider as False values.
            downcast_float: If True, downcast float64 to float32 during conversion.
            random_state: A seed for the random number generator.
            num_workers: Number of workers to use for parallel processing.
            cache_size: Maximum number of cache entries for the LRUCacheMixin.

        Warning:
            The `forced_numerical_columns`, `forced_categorical_columns`, `forced_boolean_columns`, and `drop_columns`
            parameters are mutually exclusive. Meaning, a column cannot be specified in more than one of these
            parameters. If a column is specified in more than one of these parameters, the last parameter will be used
            and the previous ones will be ignored.

        Example:
            >>> import vaex
            >>> from mleko.dataset.convert import CSVToArrowConverter
            >>> converter = CSVToArrowConverter(
            ...     output_directory="cache",
            ...     forced_numerical_columns=["x"],
            ...     forced_categorical_columns=["y"],
            ...     forced_boolean_columns=["z"],
            ...     drop_columns=["a"],
            ...     na_values=["-9999"],
            ...     true_values=["t"],
            ...     false_values=["f"],
            ...     downcast_float=True,
            ...     random_state=42,
            ...     num_workers=4,
            ...     cache_size=1,
            ... )
            >>> df = converter.convert(["data.csv"])
        """
        super().__init__(output_directory, cache_size)
        self._forced_numerical_columns = tuple(forced_numerical_columns)
        self._forced_categorical_columns = tuple(forced_categorical_columns)
        self._forced_boolean_columns = tuple(forced_boolean_columns)
        self._drop_columns = tuple(drop_columns)
        self._na_values = tuple(na_values)
        self._true_values = tuple(true_values)
        self._false_values = tuple(false_values)
        self._downcast_float = downcast_float
        self._num_workers = num_workers
        self._random_state = random_state

    def convert(self, file_paths: list[Path] | list[str], force_recompute: bool = False) -> vaex.DataFrame:
        """Converts a list of CSV files to Arrow format and returns a `vaex` dataframe joined from the converted data.

        The method takes care of caching, and results will be reused accordingly unless `force_recompute`
        is set to True. The resulting dataframe is a `vaex` DataFrame joined from the converted data.
        The conversion is done in chunks to optimize parallel processing.

        Note:
            Will read the first `100,000/len(file_paths)` rows of each file to determine if the file is the same as the
            one in the cache. If the file is the same, the cache will be used. Otherwise, the file will be converted
            and the cache will be updated.

        Args:
            file_paths: A list of file paths to be converted.
            force_recompute: If set to True, forces recomputation and ignores the cache.

        Returns:
            The resulting dataframe with the combined converted data.
        """
        return self._cached_execute(
            lambda_func=lambda: self._convert(file_paths),
            cache_keys=[
                self._forced_numerical_columns,
                self._forced_categorical_columns,
                self._forced_boolean_columns,
                self._drop_columns,
                self._na_values,
                self._true_values,
                self._false_values,
                self._downcast_float,
                (file_paths, CSVFingerprinter(n_rows=100_000 // len(file_paths))),
            ],
            force_recompute=force_recompute,
        )

    @staticmethod
    def _convert_csv_file_to_arrow(
        file_path: Path | str,
        output_directory: Path,
        dataframe_suffix: str,
        forced_numerical_columns: tuple[str, ...],
        forced_categorical_columns: tuple[str, ...],
        forced_boolean_columns: tuple[str, ...],
        drop_columns: tuple[str, ...],
        na_values: tuple[str, ...],
        true_values: tuple[str, ...],
        false_values: tuple[str, ...],
        downcast_float: bool,
    ) -> None:
        """Converts a single CSV file to Arrow format using the provided options and saves it to the output directory.

        This operation is done in chunks to optimize parallel processing. The resulting dataframe is saved in the
        output directory with the given suffix.

        Args:
            file_path: The path of the CSV file to be converted.
            output_directory: The directory where the converted file should be saved.
            dataframe_suffix: The suffix for the converted dataframe files.
            forced_numerical_columns: A sequence of column names to be forced to numerical type.
            forced_categorical_columns: A sequence of column names to be forced to categorical type.
            forced_boolean_columns: A sequence of column names to be forced to boolean type.
            drop_columns: A sequence of column names to be dropped from the dataframe.
            na_values: A sequence of values to be considered as NaN.
            true_values: A sequence of values to be considered as True.
            false_values: A sequence of values to be considered as False.
            downcast_float: If set to True, downcasts float64 to float32.
        """
        file_path = Path(file_path)

        float_type = "float64"
        if downcast_float:
            float_type = "float32"

        dtypes = {}
        for col in forced_numerical_columns:
            dtypes[col] = float_type
        for col in forced_categorical_columns:
            dtypes[col] = "string"
        for col in forced_boolean_columns:
            dtypes[col] = "boolean"

        df_chunk = vaex.from_csv_arrow(
            file_path,
            read_options=arrow_csv.ReadOptions(use_threads=True),
            convert_options=arrow_csv.ConvertOptions(
                column_types=dtypes,
                null_values=na_values,
                true_values=true_values,
                false_values=false_values,
                strings_can_be_null=True,
                quoted_strings_can_be_null=True,
                timestamp_parsers=[
                    arrow_csv.ISO8601,
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                ],
            ),
        ).drop(drop_columns)

        output_path = output_directory / f"df_chunk_{file_path.stem}.{dataframe_suffix}"
        df_chunk.export(output_path)
        df_chunk.close()

    def _convert(self, file_paths: list[Path] | list[str]) -> vaex.DataFrame:
        """Converts a list of CSV files to Arrow format using parallel processing.

        Chunks of files are processed in parallel and saved in the output directory.

        Args:
            file_paths: A list of file paths to be converted.

        Returns:
            A DataFrame containing the merged chunks.
        """
        with tqdm(total=len(file_paths), desc="Converting CSV files") as pbar:
            with futures.ProcessPoolExecutor(max_workers=min(self._num_workers, len(file_paths))) as executor:
                for _ in executor.map(
                    CSVToVaexConverter._convert_csv_file_to_arrow,
                    file_paths,
                    repeat(self._cache_directory),
                    repeat(self._cache_file_suffix),
                    repeat(self._forced_numerical_columns),
                    repeat(self._forced_categorical_columns),
                    repeat(self._forced_boolean_columns),
                    repeat(self._drop_columns),
                    repeat(self._na_values),
                    repeat(self._true_values),
                    repeat(self._false_values),
                    repeat(self._downcast_float),
                ):
                    pbar.update(1)

        return vaex.open(self._cache_directory / f"df_chunk_*.{self._cache_file_suffix}")

    def _write_cache_file(self, cache_file_path: Path, output: vaex.DataFrame) -> None:
        """Writes the results of the DataFrame conversion to Arrow format in a cache file with arrow suffix.

        Args:
            cache_file_path: The path of the cache file to be written.
            output: The Vaex DataFrame to be saved in the cache file.
        """
        super()._write_cache_file(cache_file_path, output)
        clear_directory(cache_file_path.parent, pattern=f"df_chunk_*.{self._cache_file_suffix}")