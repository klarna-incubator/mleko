"""Module for the frequency encoder transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable, Literal

import vaex
import vaex.ml

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class FrequencyEncoderTransformer(BaseTransformer):
    """Transforms features using frequency encoding."""

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...],
        unseen_strategy: Literal["zero", "nan"] = "nan",
        cache_size: int = 1,
    ) -> None:
        """Initializes the transformer.

        Uses the `vaex.ml.FrequencyEncoder` transformer, which encodes categorical features using the frequency of
        their respective samples. If a value is not seen during fitting, it will be encoded as zero or nan,
        depending on the `unseen_strategy` parameter. Missing values will be encoded as nan, but will still count
        towards the frequency of other values.

        Warning:
            Should only be used with categorical features. High cardinality features are not recommended as they will
            result in very small frequencies.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the transformer.
            unseen_strategy: Strategy to use for unseen values once the transformer is fitted.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import FrequencyEncoderTransformer
            >>> from mleko.utils.vaex_helpers import get_column
            >>> df = vaex.from_arrays(
            ...     a=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ...     b=["a", "a", "a", "a", None, None, None, None, None, None],
            ...     c=["a", "b", "b", "b", "b", "b", None, None, None, None],
            ... )
            >>> df = FrequencyEncoderTransformer(
            ...     cache_directory=".",
            ...     features=["a", "b"],
            ... ).transform(df)
            >>> df["a"].tolist()
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            >>> df["b"].tolist()
            [0.4, 0.4, 0.4, 0.4, nan, nan, nan, nan, nan, nan]
        """
        super().__init__(cache_directory, features, cache_size)
        self._unseen_strategy = unseen_strategy

    def transform(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Transforms the features in the DataFrame using frequency encoding.

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
        """Transforms the features in the DataFrame using frequency encoding.

        Args:
            dataframe: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """
        frequency_encoder = vaex.ml.FrequencyEncoder(
            features=self._features, unseen_strategy=self._unseen_strategy, prefix=""
        )
        logger.info(f"Transforming features using frequency encoding ({len(self._features)}): {self._features}.")
        transformed_df = frequency_encoder.fit_transform(dataframe)
        return transformed_df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Append the `unseen_strategy` to the fingerprint.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), self._unseen_strategy
