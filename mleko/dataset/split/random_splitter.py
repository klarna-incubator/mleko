"""The module provides a `RandomSplitter` class for splitting `vaex` DataFrames randomly.

The splitter can be used to split a `vaex` DataFrame into two parts, with the split being performed randomly. The split
can be stratified by specifying a column name to use for stratification.
"""
from __future__ import annotations

from pathlib import Path

import vaex
from sklearn.model_selection import train_test_split

from mleko.cache.fingerprinters import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_filtered_df

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
        cache_directory: str | Path,
        data_split: tuple[float, float] = (0.80, 0.20),
        shuffle: bool = True,
        stratify: str | None = None,
        random_state: int | None = None,
        cache_size: int = 1,
    ):
        """Initializes the `RandomSplitter` with the given parameters.

        Note:
            The stratification is performed before the split, meaning that the split will be performed on the stratified
            data. For example, if the data is split into 80% train and 20% test, and the stratification column contains
            80% of the rows with value 0 and 20% of the rows with value 1, the resulting split will contain 80% of the
            rows with value 0 and 20% of the rows with value 1.

        Args:
            cache_directory: The target directory where the split dataframes are to be saved.
            data_split: A tuple containing the desired split percentages or weights for the train and test dataframes.
                If the sum of the values is not equal to 1, the values will be normalized. Meaning, if the values are
                (0.90, 0.20), the resulting split will be (0.818, 0.182).
            shuffle: Whether to shuffle the data before splitting.
            stratify: The name of the column to use for stratification. If None, stratification will not be performed.
            random_state: The seed to use for random number generation.
            cache_size: The maximum number of entries to keep in the cache.

        Example:
            >>> import vaex
            >>> from mleko.data.split import RandomSplitter
            >>> df = vaex.from_arrays(x=[1, 2, 3, 4], y=[0, 1, 1, 0])
            >>> splitter = RandomSplitter(cache_directory="cache", data_split=(0.50, 0.50), shuffle=True, stratify="y")
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
        self._stratify = stratify
        self._random_state = random_state

    def split(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        Splits the dataframe into train and test sets according to the proportions, shuffle,
        stratification, and random state provided during initializing the splitter. Will read from the cache if
        available, unless `force_recompute=True`.

        Args:
            dataframe: The dataframe to be split.
            force_recompute: Whether to force recompute the split, even if the cache is available.

        Returns:
            A tuple containing the split dataframes.
        """
        return self._cached_execute(  # type: ignore
            lambda_func=lambda: self._split(dataframe),
            cache_keys=[
                self._idx2_size,
                self._shuffle,
                self._stratify,
                self._random_state,
                (dataframe, VaexFingerprinter()),
            ],
            force_recompute=force_recompute,
        )

    def _split(self, dataframe: vaex.DataFrame) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        Args:
            dataframe: The dataframe to be split.

        Returns:
            A tuple containing the split dataframes.
        """
        index_name = "index"
        dataframe[index_name] = vaex.vrange(0, dataframe.shape[0])
        index = get_column(dataframe, index_name)
        target = get_column(dataframe, self._stratify).to_numpy() if self._stratify else None

        if self._shuffle:
            logger.info("Shuffling data before splitting.")
        if target is None:
            logger.info("Splitting data without stratification.")
        else:
            logger.info(f"Splitting data with stratification on column {self._stratify!r}.")

        idx1, idx2 = train_test_split(
            index.values,
            test_size=self._idx2_size,
            random_state=self._random_state,
            shuffle=self._shuffle,
            stratify=target,
        )

        df1 = get_filtered_df(dataframe, index.isin(idx1)).extract()
        df2 = get_filtered_df(dataframe, index.isin(idx2)).extract()
        logger.info(f"Split dataframe into two dataframes with shapes {df1.shape} and {df2.shape}.")
        df1.delete_virtual_column(index_name)
        df2.delete_virtual_column(index_name)
        return df1, df2
