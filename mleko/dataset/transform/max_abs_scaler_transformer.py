"""Module for the max-abs scaler transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
import vaex.ml

from mleko.dataset.data_schema import DataSchema
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
        features: list[str] | tuple[str, ...],
        cache_directory: str | Path = "data/max-abs-scaler-transformer",
        cache_size: int = 1,
    ) -> None:
        """Initializes the max absolute scaler transformer.

        The max absolute scaler transformer will scale each feature by its maximum absolute value. This transformer
        will not shift/center the data, and thus will not destroy any sparsity.

        Warning:
            Should only be used with numerical features. There should be no missing values in the features
            or an error will be raised.

        Args:
            features: List of feature names to be used by the transformer.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import MaxAbsScalerTransformer
            >>> df = vaex.from_arrays(
            ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     b=[-1, -2, -3, -4, -5, 0, 1, 2, 3, 4]
            ... )
            >>> ds = DataSchema(
            ...     numerical=["a", "b"],
            ... )
            >>> _, _, df = MaxAbsScalerTransformer(
            ...     features=["a", "b"],
            ... ).fit_transform(ds, df)
            >>> df["a"].tolist()
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            >>> df["b"].tolist()
            [-0.2, -0.4, -0.6, -0.8, -1.0, 0.0, 0.2, 0.4, 0.6, 0.8]
        """
        super().__init__(features, cache_directory, cache_size)
        self._transformer = vaex.ml.MaxAbsScaler(features=self._features, prefix="")

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.ml.MaxAbsScaler]:
        """Fits the transformer on the given DataFrame.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to fit the transformer on.

        Returns:
            Updated data schema and fitted transformer.
        """
        logger.info(f"Fitting max-abs scaler transformer ({len(self._features)}): {self._features}.")
        self._transformer.fit(dataframe)

        return data_schema, self._transformer

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the features in the DataFrame using max-abs scaling.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            Updated data schema and transformed DataFrame.
        """
        logger.info(f"Transforming features using max-abs scaling ({len(self._features)}): {self._features}.")
        transformed_df = self._transformer.transform(dataframe)

        return data_schema, transformed_df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint()
