"""The module provides `ExpressionSplitter` class for splitting Vaex DataFrames based on a given expression."""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.cache.fingerprinters import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_splitter import BaseSplitter


logger = CustomLogger()
"""A module-level logger instance."""


class ExpressionSplitter(BaseSplitter):
    """A class that handles splitting of `vaex` DataFrames based on a given `vaex` expression."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        expression: str,
        cache_size: int = 1,
    ):
        """Initializes the `ExpressionSplitter` with the given parameters.

        The expression should be a valid Vaex expression that evaluates to a boolean
        value. The rows for which the expression evaluates to True will be returned as the first dataframe,
        and the remaining rows will be returned as the second dataframe.

        Note:
            To filter by a date column, use the `scalar_datetime` function. For example, to filter by a date column
            named `date` and return the rows before `2020-06-01`, use the
            expression `"date < scalar_datetime('2020-06-01')"`.

        Args:
            cache_directory: The target directory where the split dataframes are to be saved.
            expression: A valid Vaex expression that evaluates to a boolean value. The rows for which the expression
                evaluates to True will be returned as the first dataframe, and the remaining rows will be returned
                as the second dataframe.
            cache_size: The maximum number of entries to keep in the cache.

        Example:
            >>> import vaex
            >>> from mleko.data.split import ExpressionSplitter
            >>> df = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6])
            >>> splitter = ExpressionSplitter(cache_directory="cache", expression="x > 1")
            >>> df_train, df_test = splitter.split(df)
            >>> df_train
                #    x    y
                0    2    5
                1    3    6
            >>> df_test
                #    x    y
                0    1    4
        """
        super().__init__(cache_directory, cache_size)
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
        """Split the given dataframe into two parts.

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
