"""Module for the label encoder transformer."""
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
            cache_directory: Directory where the resulting DataFrame will be stored locally.
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
            >>> df = LabelEncoderTransformer(
            ...     cache_directory=".",
            ...     features=["a", "b"],
            ...     allow_unseen=True,
            ... ).transform(df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        """
        super().__init__(cache_directory, features, cache_size)
        self._allow_unseen = allow_unseen

    def transform(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Transforms the features of the given DataFrame using label encoding.

        Will cache the resulting DataFrame in the cache directory.

        Args:
            dataframe: The DataFrame to transform.
            force_recompute: Whether to force recomputation of the transformation.

        Returns:
            The transformed DataFrame.
        """
        return self._cached_execute(
            lambda_func=lambda: self._transform(dataframe),
            cache_keys=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            force_recompute=force_recompute,
        )

    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the features of the given DataFrame using label encoding.

        Args:
            dataframe: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """
        label_encoder = vaex.ml.LabelEncoder(features=self._features, prefix="")
        logger.info(f"Transforming features using label encoding ({len(self._features)}): {self._features}.")
        transformed_df = label_encoder.fit_transform(dataframe)
        return transformed_df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Appends the `allow_unseen` parameter to the fingerprint of the base class.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), self._allow_unseen
