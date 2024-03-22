"""This module contains the `ExpressionFilter` class, which is used to filter `vaex` DataFrames."""

from __future__ import annotations

from pathlib import Path

import vaex

from mleko.cache.fingerprinters import DictFingerprinter, VaexFingerprinter
from mleko.cache.handlers.vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER
from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_filter import BaseFilter


logger = CustomLogger()
"""A module-level logger instance."""


class ExpressionFilter(BaseFilter):
    """A class that handles filtering of `vaex` DataFrames based on a given `vaex` expression."""

    @auto_repr
    def __init__(
        self,
        expression: str,
        cache_directory: str | Path = "data/expression-filter",
        cache_size: int = 1,
    ) -> None:
        """Initializes the `ExpressionFilter` with the given expression.

        The expression should be a valid Vaex expression that evaluates to a boolean value.

        Note:
            To filter by a date column, use the `scalar_datetime` function. For example, to filter by a date column
            named `date` and return the rows before `2020-06-01`, use the
            expression `"date < scalar_datetime('2020-06-01')"`.

        Args:
            expression: The expression to be used for filtering.
            cache_directory: The target directory where the filtered dataframes are to be saved.
            cache_size: The maximum number of cache entries.

        Example:
            >>> import vaex
            >>> from mleko.data.filter import ExpressionFilter
            >>> df = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6])
            >>> filter = ExpressionFilter(expression="x > 1")
            >>> df_filtered = filter.filter(df)
            >>> df_filtered
                #    x    y
                0    2    5
                1    3    6
        """
        super().__init__(cache_directory, cache_size)
        self._expression = expression

    def filter(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> vaex.DataFrame:
        """Filter the given dataframe based on the expression.

        Args:
            data_schema: The data schema to be used for filtering.
            dataframe: The dataframe to be filtered.
            cache_group: The cache group to use.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.
            disable_cache: If set to True, disables the cache.

        Returns:
            The filtered dataframe.
        """
        return self._cached_execute(
            lambda_func=lambda: self._filter(data_schema, dataframe),
            cache_key_inputs=[
                self._expression,
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=VAEX_DATAFRAME_CACHE_HANDLER,
            disable_cache=disable_cache,
        )

    def _filter(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Filter the given dataframe based on the expression.

        Args:
            dataframe: The dataframe to be filtered.

        Returns:
            The filtered dataframe.
        """
        logger.info(f"Filtering dataframe based on expression {self._expression!r}.")
        filtered_df = dataframe.filter(f"({self._expression})").extract()
        logger.info(
            f"Filtered dataframe into shape {filtered_df.shape}, "
            f"dropped {dataframe.shape[0] - filtered_df.shape[0]} rows."
        )
        return filtered_df
