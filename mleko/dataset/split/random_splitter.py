"""The module provides a `RandomSplitter` class for splitting `vaex` DataFrames randomly.

The splitter can be used to split a `vaex` DataFrame into two parts, with the split being performed randomly. The split
can be stratified by specifying a column name to use for stratification.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import vaex
from sklearn.model_selection import train_test_split

from mleko.cache.fingerprinters import VaexFingerprinter
from mleko.cache.handlers.vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER
from mleko.utils import auto_repr, get_column, get_columns, get_filtered_df
from mleko.utils.custom_logger import CustomLogger

from .base_splitter import BaseSplitter


logger = CustomLogger()
"""A CustomLogger instance that's used throughout the module for logging."""


class RandomSplitter(BaseSplitter):
    """A class that handles random splitting of `vaex` DataFrames.

    This class provides a method for splitting a `vaex` DataFrame into two parts, with the split being performed
    randomly. The split can be stratified by specifying a column name to use for stratification.
    """

    @auto_repr
    def __init__(
        self,
        data_split: tuple[float, float] = (0.80, 0.20),
        shuffle: bool = True,
        stratify: str | tuple[str, ...] | list[str] | None = None,
        random_state: int | None = 42,
        cache_directory: str | Path = "data/random-splitter",
        cache_size: int = 1,
    ) -> None:
        """Initializes the `RandomSplitter` with the given parameters.

        Note:
            If `stratify` is not None and the type of the data is string/categorical the feature (s) must not have
            any missing values. Please make sure to handle missing values before using this splitter by either
            dropping the rows with missing values, imputing the missing values, or transforming the target to
            numeric or boolean.

            If multiple features are provided for stratification, there must be at least one row for each unique
            combination of the feature values. Otherwise, the splitter will raise an error.

        Args:
            data_split: A tuple containing the desired split percentages or weights for the train and test dataframes.
                If the sum of the values is not equal to 1, the values will be normalized. Meaning, if the values are
                (0.90, 0.20), the resulting split will be (0.818, 0.182).
            shuffle: Whether to shuffle the data before splitting.
            stratify: Name of the feature(s) to use for stratification. If None, the data will be split without
                stratification. If a list of features is provided, the data will be stratified based on the unique
                combinations of the features, i.e., the data will be split such that each unique combination of the
                features is present in both splits in the same proportion as in the original data.
            random_state: The seed to use for random number generation.
            cache_directory: The target directory where the split dataframes are to be saved.
            cache_size: The maximum number of entries to keep in the cache.

        Example:
            >>> import vaex
            >>> from mleko.data.split import RandomSplitter
            >>> df = vaex.from_arrays(x=[1, 2, 3, 4], y=[0, 1, 1, 0])
            >>> splitter = RandomSplitter(data_split=(0.50, 0.50), shuffle=True, stratify="y")
            >>> df_train, df_test = splitter.split(df)
            >>> df_train
                #    x    y
                0    1    0
                1    3    1
            >>> df_test
                #    x    y
                0    2    1
                1    4    0
        """
        super().__init__(cache_directory, cache_size)
        self._idx2_size = [split / sum(data_split) for split in data_split][1]
        self._shuffle = shuffle
        self._stratify = tuple(stratify) if isinstance(stratify, (list, tuple)) else (stratify,) if stratify else None
        self._random_state = random_state

    def split(
        self,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        Splits the dataframe into train and test sets according to the proportions, shuffle,
        stratification, and random state provided during initializing the splitter. Will read from the cache if
        available, unless `force_recompute=True`.

        Args:
            dataframe: The dataframe to be split.
            cache_group: The cache group to use.
            force_recompute: Whether to force recompute the split, even if the cache is available.
            disable_cache: If set to True, disables the cache.

        Returns:
            A tuple containing the split dataframes.
        """
        return self._cached_execute(
            lambda_func=lambda: self._split(dataframe),
            cache_key_inputs=[
                self._idx2_size,
                self._shuffle,
                self._stratify,
                self._random_state,
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=VAEX_DATAFRAME_CACHE_HANDLER,
            disable_cache=disable_cache,
        )

    def _split(self, dataframe: vaex.DataFrame) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        Args:
            dataframe: The dataframe to be split.

        Returns:
            A tuple containing the split dataframes.
        """
        index_name = "index"
        df = dataframe.copy()
        df[index_name] = vaex.vrange(0, df.shape[0])
        index = get_column(df, index_name)
        stratification: pd.DataFrame | None = (
            get_columns(df, list(self._stratify)).to_pandas_df() if self._stratify else None  # type: ignore
        )

        if self._shuffle:
            logger.info("Shuffling data before splitting.")
        if stratification is None:
            logger.info("Splitting data without stratification.")
        else:
            logger.info(f"Splitting data with stratification on feature(s) {self._stratify!r}.")

        idx1, idx2 = train_test_split(
            index.values,
            test_size=self._idx2_size,
            random_state=self._random_state,
            shuffle=self._shuffle,
            stratify=stratification,
        )

        df1 = get_filtered_df(df, index.isin(idx1)).extract()
        df2 = get_filtered_df(df, index.isin(idx2)).extract()
        logger.info(f"Split dataframe into two dataframes with shapes {df1.shape} and {df2.shape}.")
        df1.delete_virtual_column(index_name)
        df2.delete_virtual_column(index_name)
        return df1, df2
