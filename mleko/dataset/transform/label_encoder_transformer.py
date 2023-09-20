"""Module for the label encoder transformer."""
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


class LabelEncoderTransformer(BaseTransformer):
    """Transforms features using label encoding."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        allow_unseen: bool = True,
        cache_size: int = 1,
    ) -> None:
        """Initializes the transformer.

        Uses the `vaex.ml.LabelEncoder` transformer, which encodes categorical features with integer values between 0
        and n_classes-1. If a value is not seen during fitting, it will be encoded as -1, unless `allow_unseen` is
        set to False, in which case an error will be raised.

        Warning:
            Should only be used with categorical features.

        Args:
            cache_directory: Directory where the cache will be stored locally.
            features: List of feature names to be used by the transformer.
            allow_unseen: Whether to allow unseen values once the transformer is fitted.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import LabelEncoderTransformer
            >>> df = vaex.from_arrays(
            ...     a=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ...     b=["a", "a", "a", "a", None, None, None, None, None, None],
            ...     c=["a", "b", "b", "b", "b", "b", None, None, None, None],
            ... )
            >>> ds = DataSchema(
            ...     categorical=["a", "b", "c"],
            ... )
            >>> _, _, df = LabelEncoderTransformer(
            ...     cache_directory=".",
            ...     features=["a", "b"],
            ...     allow_unseen=True,
            ... ).fit_transform(ds, df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        """
        super().__init__(cache_directory, features, cache_size)
        self._allow_unseen = allow_unseen
        self._transformer = vaex.ml.LabelEncoder(allow_unseen=self._allow_unseen, features=self._features, prefix="")

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.ml.LabelEncoder]:
        """Fits the transformer on the given DataFrame.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to fit the transformer on.

        Returns:
            Updated data schema and fitted transformer.
        """
        logger.info(f"Fitting label encoder transformer ({len(self._features)}): {self._features}.")
        self._transformer.fit(dataframe)

        return data_schema, self._transformer

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the features of the given DataFrame using label encoding.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            Updated data schema and transformed DataFrame.
        """
        logger.info(f"Transforming features using label encoding ({len(self._features)}): {self._features}.")
        transformed_df = self._transformer.transform(dataframe)

        return data_schema, transformed_df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Appends the `allow_unseen` parameter to the fingerprint of the base class.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), self._allow_unseen
