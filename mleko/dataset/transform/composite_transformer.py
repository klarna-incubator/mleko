"""Module for the composite transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class CompositeTransformer(BaseTransformer):
    """A transformer that combines multiple transformers.

    It is possible to combine multiple transformers into a single transformer. This can be useful when multiple
    transformers need to be applied to a DataFrame and storing the intermediate DataFrames is not desired.
    """

    @auto_repr
    def __init__(
        self,
        cache_directory: str | Path,
        transformers: list[BaseTransformer] | tuple[BaseTransformer, ...],
        cache_size: int = 1,
    ) -> None:
        """Initializes the composite transformer.

        The composite transformer will combine the transformers into a single transformer. Each transformer will be
        applied to the DataFrame in the order they are specified.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            transformers: List of transformers to be combined.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import (
            ...     CompositeTransformer,
            ...     LabelEncoderTransformer,
            ...     FrequencyEncoderTransformer
            ... )
            >>> df = vaex.from_arrays(
            ...     a=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ...     b=["a", "a", "a", "a", None, None, None, None, None, None],
            ... )
            >>> transformer = CompositeTransformer(
            ...     cache_directory=".",
            ...     transformers=[
            ...         LabelEncoderTransformer(
            ...             cache_directory=".",
            ...             features=["a"],
            ...         ),
            ...         FrequencyEncoderTransformer(
            ...             cache_directory=".",
            ...             features=["b"],
            ...         ),
            ...     ],
            ... )
            >>> df = transformer.transform(df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [0.4, 0.4, 0.4, 0.4, nan, nan, nan, nan, nan, nan]
        """
        super().__init__(cache_directory, [], cache_size)
        self._transformers = tuple(transformers)

    def transform(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Transforms the DataFrame using the transformers in the order they are specified.

        Args:
            dataframe: The DataFrame to transform.
            force_recompute: Whether to force the recomputation of the transformation.

        Returns:
            The transformed DataFrame.
        """
        return self._cached_execute(
            lambda_func=lambda: self._transform(dataframe),
            cache_keys=[self._fingerprint(), (dataframe, VaexFingerprinter())],
            force_recompute=force_recompute,
        )

    def _transform(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Returns the transformed DataFrame.

        Args:
            dataframe: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """
        for i, transformer in enumerate(self._transformers):
            logger.info(
                f"Executing composite feature transformation step {i+1}/{len(self._transformers)}: "
                f"{transformer.__class__.__name__}."
            )
            dataframe = transformer._transform(dataframe).extract()
            logger.info(f"Finished composite transformation step {i+1}/{len(self._transformers)}.")
        return dataframe

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), tuple(transformer._fingerprint() for transformer in self._transformers)
