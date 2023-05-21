"""A module that provides classes for splitting Vaex DataFrames.

This module provides classes for splitting Vaex DataFrames into two parts. The split can be performed randomly or
based on a given expression.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex
from sklearn.model_selection import train_test_split

from mleko.cache.cache import LRUCacheMixin, VaexArrowCacheFormatMixin
from mleko.cache.fingerprinters import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex import get_column, get_columns


logger = CustomLogger()
"""A CustomLogger instance that's used throughout the module for logging."""


class BaseDataSplitter(ABC):
    """Abstract base class for data splitters.

    Provides a common interface for splitting a dataframe into two parts by implementing the `split` method.
    """

    def __init__(self, output_directory: str | Path):
        """Initializes the BaseDataSplitter with an output directory.

        Args:
            output_directory: The target directory where the split dataframes are to be saved.
        """
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def split(self, dataframe: vaex.DataFrame) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        The implementation of this method should split the given dataframe into two parts and return them as a tuple.

        Args:
            dataframe: The dataframe to be split.

        Returns:
            A tuple containing the split dataframes.
        """
        raise NotImplementedError


class RandomDataSplitter(BaseDataSplitter, VaexArrowCacheFormatMixin, LRUCacheMixin):
    """A class that handles random splitting of Vaex DataFrames.

    This class provides a method for splitting a Vaex DataFrame into two parts, with the split being performed
    randomly. The split can be stratified by specifying a column name to use for stratification.
    """

    @auto_repr
    def __init__(
        self,
        output_directory: str | Path,
        data_split: tuple[float, float] = (0.80, 0.20),
        shuffle: bool = True,
        stratify: str | None = None,
        random_state: int | None = None,
        max_cache_entries: int = 1,
    ):
        """Initializes the RandomDataSplitter with the given parameters.

        Args:
            output_directory: The target directory where the split dataframes are to be saved.
            data_split: A tuple containing the desired split percentages or weights for the train and test dataframes.
                If the sum of the values is not equal to 1, the values will be normalized. Meaning, if the values are
                (0.90, 0.20), the resulting split will be (0.818, 0.182).
            shuffle: Whether to shuffle the data before splitting.
            stratify: The name of the column to use for stratification. If None, stratification will not be performed.
            random_state: The seed to use for random number generation.
            max_cache_entries: The maximum number of entries to keep in the cache.
        """
        BaseDataSplitter.__init__(self, output_directory)
        VaexArrowCacheFormatMixin.__init__(self)
        LRUCacheMixin.__init__(self, output_directory, VaexArrowCacheFormatMixin.cache_file_suffix, max_cache_entries)
        self._test_size = [split / sum(data_split) for split in data_split][1]
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
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.

        Returns:
            A tuple containing the split dataframes.
        """
        return self._cached_execute(  # type: ignore
            lambda_func=lambda: self._split(dataframe),
            cache_keys=[
                self._test_size,
                self._shuffle,
                self._stratify,
                self._random_state,
                (dataframe, VaexFingerprinter()),
            ],
            force_recompute=force_recompute,
        )

    def _split(self, dataframe: vaex.DataFrame) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Perform the actual splitting of the dataframe.

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

        train_idx, test_idx = train_test_split(
            index.values,
            test_size=self._test_size,
            random_state=self._random_state,
            shuffle=self._shuffle,
            stratify=target,
        )

        df_train = get_columns(dataframe, index.isin(train_idx)).extract()
        df_test = get_columns(dataframe, index.isin(test_idx)).extract()
        logger.info(f"Split dataframe into two dataframes with shapes {df_train.shape} and {df_test.shape}.")
        df_train.delete_virtual_column(index_name)
        df_test.delete_virtual_column(index_name)
        return df_train, df_test


class ExpressionDataSplitter(BaseDataSplitter, VaexArrowCacheFormatMixin, LRUCacheMixin):
    """A class that handles splitting of Vaex DataFrames based on a given expression.

    This class provides a method for splitting a Vaex DataFrame into two parts, with the split being performed
    based on a given expression. The expression should be a valid Vaex expression that evaluates to a boolean
    value. The split can be stratified by specifying a column name to use for stratification. The rows for which
    the expression evaluates to True will be returned as the first dataframe, and the remaining rows will be
    returned as the second dataframe.

    Note:
        To filter by a date column, use the `scalar_datetime` function. For example, to filter by a date column
        named `date` and return the rows before 2020-06-01, use the expression `date < scalar_datetime("2020-06-01")`.

    Example:
        >>> import vaex
        >>> from mleko.data.splitters import ExpressionDataSplitter
        >>> df = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6])
        >>> splitter = ExpressionDataSplitter(output_directory="cache", expression="x > 1")
        >>> df_train, df_test = splitter.split(df)
        >>> df_train
            #    x    y
            0    2    5
            1    3    6
        >>> df_test
            #    x    y
            0    1    4
    """

    @auto_repr
    def __init__(
        self,
        output_directory: str | Path,
        expression: str,
        max_cache_entries: int = 1,
    ):
        """Initializes the ExpressionDataSplitter with the given parameters.

        Args:
            output_directory: The target directory where the split dataframes are to be saved.
            expression: A valid Vaex expression that evaluates to a boolean value. The rows for which the expression
                evaluates to True will be returned as the first dataframe, and the remaining rows will be returned
                as the second dataframe.
            max_cache_entries: The maximum number of entries to keep in the cache.
        """
        BaseDataSplitter.__init__(self, output_directory)
        VaexArrowCacheFormatMixin.__init__(self)
        LRUCacheMixin.__init__(self, output_directory, VaexArrowCacheFormatMixin.cache_file_suffix, max_cache_entries)
        self._expression = expression

    def split(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Split the given dataframe into two parts.

        Args:
            dataframe: The dataframe to be split.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.

        Returns:
            A tuple containing the split dataframes.
        """
        return self._cached_execute(  # type: ignore
            lambda_func=lambda: self._split(dataframe),
            cache_keys=[
                self._expression,
                (dataframe, VaexFingerprinter()),
            ],
            force_recompute=force_recompute,
        )

    def _split(self, dataframe: vaex.DataFrame) -> tuple[vaex.DataFrame, vaex.DataFrame]:
        """Perform the actual splitting of the dataframe.

        Args:
            dataframe: The dataframe to be split.

        Returns:
            A tuple containing the split dataframes.
        """
        logger.info(f"Splitting dataframe based on expression {self._expression!r}.")
        filtered_df = dataframe.filter(f"({self._expression})").extract()
        remainder_df = dataframe.filter(f"~({self._expression})").extract()
        logger.info(f"Split dataframe into two dataframes with shapes {filtered_df.shape} and {remainder_df.shape}.")
        return filtered_df, remainder_df
