"""Module for the composite transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable

import vaex

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
        applied to the DataFrame in the order they are specified. Caching of the intermediate DataFrames is disabled
        and will only be performed on the final DataFrame.

        Args:
            cache_directory: Directory where the cache will be stored locally.
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
            >>> _, df = transformer.fit_transform(df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [0.4, 0.4, 0.4, 0.4, nan, nan, nan, nan, nan, nan]
        """
        super().__init__(cache_directory, [], cache_size)
        self._transformers = tuple(transformers)
        self._transformer: list[Any] = []

    def _fit(self, dataframe: vaex.DataFrame) -> list[Any]:
        """Fits the transformer to the specified DataFrame.

        Args:
            dataframe: DataFrame to be fitted.

        Returns:
            List of fitted transformers.
        """
        fitted_transformers: list[Any] = []
        for i, transformer in enumerate(self._transformers):
            logger.info(
                f"Fitting composite feature transformation step {i+1}/{len(self._transformers)}: "
                f"{transformer.__class__.__name__}."
            )
            fitted_transformer = transformer._fit(dataframe)
            fitted_transformers.append(fitted_transformer)
            logger.info(f"Finished fitting composite transformation step {i+1}/{len(self._transformers)}.")
        return fitted_transformers

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

    def _assign_transformer(self, transformer: Any) -> None:
        """Assigns the specified transformer to the transformer attribute.

        Can be overridden by subclasses to assign the transformer using a different method.

        Args:
            transformer: Transformer to be assigned.
        """
        self._transformer = transformer
        for transformer_obj, fitted_transformer in zip(self._transformers, transformer):
            transformer_obj._transformer = fitted_transformer

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), tuple(transformer._fingerprint() for transformer in self._transformers)
