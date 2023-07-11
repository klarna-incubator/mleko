"""Module for the frequency encoder transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable, Literal

import vaex
import vaex.ml

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
        disable_cache: bool = False,
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
            disable_cache: Whether to disable caching.

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
        super().__init__(cache_directory, features, cache_size, disable_cache)
        self._unseen_strategy = unseen_strategy
        self._transformer = vaex.ml.FrequencyEncoder(
            features=self._features, unseen_strategy=self._unseen_strategy, prefix=""
        )

    def _transform(self, dataframe: vaex.DataFrame, fit: bool) -> vaex.DataFrame:
        """Transforms the features in the DataFrame using frequency encoding.

        Args:
            dataframe: The DataFrame to transform.
            fit: Whether to fit the transformer on the input data.

        Returns:
            The transformed DataFrame.
        """
        if fit:
            self._fit(dataframe)

        logger.info(f"Transforming features using frequency encoding ({len(self._features)}): {self._features}.")
        transformed_df = self._transformer.transform(dataframe)
        return transformed_df

    def _fit(self, dataframe: vaex.DataFrame) -> None:
        """Fits the transformer on the input data.

        Args:
            dataframe: The DataFrame to fit the transformer on.
        """
        logger.info(f"Fitting frequency encoder transformer ({len(self._features)}): {self._features}.")
        self._transformer.fit(dataframe)

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Append the `unseen_strategy` to the fingerprint.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), self._unseen_strategy
