"""Module for the min-max scaler transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
import vaex.ml

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class MinMaxScalerTransformer(BaseTransformer):
    """Transforms features using min-max scaling."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        min_value: float = 0.0,
        max_value: float = 1.0,
        cache_size: int = 1,
    ) -> None:
        """Initializes the min-max scaler transformer.

        The min-max scaler transformer will scale each feature to a given range. If the range is not specified,
        the range will be [0, 1]. The min-max scaler will not change the data distribution.

        Warning:
            Should only be used with numerical features. There should be no missing values in the features
            or an error will be raised.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the transformer.
            min_value: The minimum value of the range.
            max_value: The maximum value of the range.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import MaxAbsScalerTransformer
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[-1, -2, -3, -4, -5, 0, 1, 2, 3, 4]
            ... )
            >>> df = MaxAbsScalerTransformer(
            ...     cache_directory=".",
            ...     features=["a", "b"],
            ... ).transform(df)
            >>> df["a"].tolist()
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            >>> df["b"].tolist()
            [-0.2, -0.4, -0.6, -0.8, -1.0, 0.0, 0.2, 0.4, 0.6, 0.8]
        """
        super().__init__(cache_directory, features, cache_size)
        self._min_value = min_value
        self._max_value = max_value

    def transform(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Transforms the features in the DataFrame using min max scaling.

        Will cache the resulting DataFrame in the cache directory.

        Args:
            dataframe: The DataFrame to transform.
            force_recompute: Whether to force recomputing the transformation.

        Returns:
            The transformed DataFrame.
        """
        return self._cached_execute(
            lambda_func=lambda: self._transform(dataframe),
            cache_keys=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            force_recompute=force_recompute,
        )

    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the features in the DataFrame using min max scaling.

        Args:
            dataframe: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """
        max_abs_scaler = vaex.ml.MinMaxScaler(
            features=self._features, prefix="", feature_range=(self._min_value, self._max_value)
        )
        logger.info(f"Transforming features using min-max scaling ({len(self._features)}): {self._features}.")
        transformed_df = max_abs_scaler.fit_transform(dataframe)
        return transformed_df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Appends the min and max values to the fingerprint.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), self._min_value, self._max_value
