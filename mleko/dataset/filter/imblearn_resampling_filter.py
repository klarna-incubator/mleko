"""A module for filtering `vaex` DataFrames using `imblearn` sampling methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import vaex
from imblearn.under_sampling.base import BaseSampler

from mleko.cache.fingerprinters.dict_fingerprinter import DictFingerprinter
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.cache.handlers.vaex_cache_handler import VAEX_DATAFRAME_CACHE_HANDLER
from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_columns, get_indices

from .base_filter import BaseFilter


logger = CustomLogger()
"""A module-level logger instance."""


class ImblearnResamplingFilter(BaseFilter):
    """A class that handles filtering of `vaex` DataFrames using `imblearn` sampling methods."""

    @auto_repr
    def __init__(
        self,
        sampler: BaseSampler,
        target_column: str,
        random_state: int | None = 42,
        enable_logging: bool = True,
        cache_directory: str | Path = "data/imblearn-sampling-filter",
        cache_size: int = 1,
    ) -> None:
        """Initializes the `ImblearnResamplingFilter` with the given `imblearn` sampler and target column.

        The `imblearn` sampler should be a sample object that inherits from `BaseSampler`.
        For example, `imblearn.under_sampling.RandomUnderSampler`. Refer to the `imblearn` documentation for more
        information (https://imbalanced-learn.org/stable/introduction.html).

        Args:
            sampler: The `imblearn` sampler to be used for sampling.
            target_column: The target column to be used for sampling.
            random_state: The random state to be used for reproducibility, and will recursively set the random state
                of the sampler and all nested objects if set.
            enable_logging: If set to True, enables logging.
            cache_directory: The target directory where the filtered dataframes are to be saved.
            cache_size: The maximum number of cache entries.

        Examples:
            >>> import vaex
            >>> from imblearn.under_sampling import RandomUnderSampler
            >>> from mleko.data.filter import ImblearnUnderSamplingFilter
            >>> df = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6])
            >>> sampler = RandomUnderSampler()
            >>> filter = ImblearnUnderSamplingFilter(sampler=sampler, target_column="x")
            >>> df_filtered = filter.filter(df)
            >>> df_filtered
                #    x    y
                0    1    4
                1    2    5
        """
        super().__init__(cache_directory, cache_size)
        self._sampler = sampler
        self._target_column = target_column
        self._random_state = random_state
        self._reset_random_state()

    def filter(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> vaex.DataFrame:
        """Filters the given DataFrame using the specified `imblearn` sampler.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be filtered.
            cache_group: The cache group to use.
            force_recompute: Forces recomputation if True, otherwise reads from the cache if available.
            disable_cache: If set to True, disables the cache.

        Returns:
            The filtered DataFrame.
        """
        return self._cached_execute(
            lambda_func=lambda: self._filter(data_schema, dataframe),
            cache_key_inputs=[
                self._sampler.__class__.__qualname__,
                (self._sampler.get_params(deep=True), DictFingerprinter()),
                self._target_column,
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=VAEX_DATAFRAME_CACHE_HANDLER,
            disable_cache=disable_cache,
        )

    def _filter(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Filters the given DataFrame using the specified `imblearn` sampler.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be filtered.

        Returns:
            The filtered DataFrame.
        """
        resampler_name = self._sampler.__class__.__qualname__
        logger.info(f"Imblearn resampling DataFrame using {resampler_name!r}.")
        df: vaex.DataFrame = dataframe.copy()

        x = get_columns(df, data_schema.get_features()).to_pandas_df()
        y = get_column(df, self._target_column).to_numpy()
        resampled_x, resampled_y = self._sampler.fit_resample(x, y)  # type: ignore
        resampled_x_len = resampled_x.shape[0]
        original_df_len = df.shape[0]

        # Over-Sampling
        if resampled_x_len > original_df_len:
            resampled_df = vaex.from_pandas(resampled_x.iloc[original_df_len:])
            resampled_df_len = resampled_df.shape[0]

            logger.info("Ensuring that the resampled DataFrame handles NaN values correctly.")
            for col_name in data_schema.get_features():
                column = get_column(resampled_df, col_name)
                if column.countnan() > 0:
                    column_values = column.values
                    resampled_df[col_name] = pa.array(column_values, mask=np.isnan(column_values))  # type: ignore
            resampled_df[self._target_column] = resampled_y[original_df_len:]

            synthetic_column_name = "Synthetic"
            logger.info(f"Setting the {synthetic_column_name!r} column to 1 for the synthetic samples.")
            resampled_df[synthetic_column_name] = np.ones(resampled_df_len, dtype="int8")
            df[synthetic_column_name] = np.zeros(original_df_len, dtype="int8")
            resampled_df = vaex.concat([df, resampled_df])
            logger.info(f"Generated {resampled_df_len} synthetic samples.")

        # Under-Sampling
        elif resampled_x_len < original_df_len:
            resampled_df = get_indices(df, resampled_x.index)  # type: ignore
            logger.info(f"Removed {original_df_len - resampled_x_len} samples using {resampler_name!r}.")

        # No Resampling
        else:  # pragma: no cover
            resampled_df = get_indices(df, resampled_x.index)  # type: ignore
            logger.info("No resampling was performed as the DataFrame size remains the same.")

        self._reset_random_state()
        return resampled_df

    def _reset_random_state(self) -> None:
        """Resets the random state of the sampler."""
        self._sampler.set_params(
            **{
                param: self._random_state
                for param in self._sampler.get_params().keys()
                if param.endswith("random_state")
            }
        )
