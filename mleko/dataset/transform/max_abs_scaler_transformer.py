"""Module for the max-abs scaler transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
import vaex.ml

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class MaxAbsScalerTransformer(BaseTransformer):
    """Transforms features using maximum absolute scaling."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        cache_size: int = 1,
        disable_cache: bool = False,
    ) -> None:
        """Initializes the max absolute scaler transformer.

        The max absolute scaler transformer will scale each feature by its maximum absolute value. This transformer
        will not shift/center the data, and thus will not destroy any sparsity.

        Warning:
            Should only be used with numerical features. There should be no missing values in the features
            or an error will be raised.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the transformer.
            cache_size: The maximum number of entries to keep in the cache.
            disable_cache: Whether to disable caching.

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
        super().__init__(cache_directory, features, cache_size, disable_cache)
        self._transformer = vaex.ml.MaxAbsScaler(features=self._features, prefix="")

    def _transform(self, dataframe: vaex.DataFrame, fit: bool) -> vaex.DataFrame:
        """Transforms the features in the DataFrame using max-abs scaling.

        Args:
            dataframe: The DataFrame to transform.
            fit: Whether to fit the transformer on the input data.

        Returns:
            The transformed DataFrame.
        """
        if fit:
            self._fit(dataframe)

        logger.info(f"Transforming features using max-abs scaling ({len(self._features)}): {self._features}.")
        transformed_df = self._transformer.transform(dataframe)
        return transformed_df

    def _fit(self, dataframe: vaex.DataFrame) -> None:
        """Fits the transformer on the given DataFrame.

        Args:
            dataframe: The DataFrame to fit the transformer on.
        """
        logger.info(f"Fitting max-abs scaler transformer ({len(self._features)}): {self._features}.")
        self._transformer.fit(dataframe)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint()
