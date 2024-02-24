"""Module for the composite transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.dataset.data_schema import DataSchema
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
        transformers: list[BaseTransformer] | tuple[BaseTransformer, ...],
        cache_directory: str | Path = "data/composite-transformer",
        cache_size: int = 1,
    ) -> None:
        """Initializes the composite transformer.

        The composite transformer will combine the transformers into a single transformer. Each transformer will be
        applied to the DataFrame in the order they are specified. Caching of the intermediate DataFrames is disabled
        and will only be performed on the final DataFrame.

        Args:
            transformers: List of transformers to be combined.
            cache_directory: Directory where the cache will be stored locally.
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
            >>> ds = DataSchema(
            ...     categorical=["a", "b"],
            ... )
            >>> transformer = CompositeTransformer(
            ...     transformers=[
            ...         LabelEncoderTransformer(
            ...             features=["a"],
            ...         ),
            ...         FrequencyEncoderTransformer(
            ...             features=["b"],
            ...         ),
            ...     ],
            ... )
            >>> _, _, df = transformer.fit_transform(ds, df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [0.4, 0.4, 0.4, 0.4, nan, nan, nan, nan, nan, nan]
        """
        super().__init__([], cache_directory, cache_size)
        self._transformers = tuple(transformers)
        self._transformer: list[Any] = []

    def _fit(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, list[Any]]:
        """Fits the transformer to the specified DataFrame.

        Args:
            data_schema: Data schema of the DataFrame.
            dataframe: DataFrame to be fitted.

        Returns:
            Updated data schema and list of fitted transformers.
        """
        fitted_transformers: list[Any] = []
        for i, transformer in enumerate(self._transformers):
            logger.info(
                f"Fitting composite feature transformation step {i+1}/{len(self._transformers)}: "
                f"{transformer.__class__.__name__}."
            )
            data_schema, fitted_transformer = transformer.fit(data_schema, dataframe, disable_cache=True)
            fitted_transformers.append(fitted_transformer)
            logger.info(f"Finished fitting composite transformation step {i+1}/{len(self._transformers)}.")
        return data_schema, fitted_transformers

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Returns the updated data schema transformed DataFrame.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            Updated data schema and transformed DataFrame.
        """
        for i, transformer in enumerate(self._transformers):
            logger.info(
                f"Executing composite feature transformation step {i+1}/{len(self._transformers)}: "
                f"{transformer.__class__.__name__}."
            )
            data_schema, dataframe = transformer.transform(data_schema, dataframe, disable_cache=True)
            dataframe = dataframe.extract()
            logger.info(f"Finished composite transformation step {i+1}/{len(self._transformers)}.")
        return data_schema, dataframe

    def _fit_transform(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, Any, vaex.DataFrame]:
        """Fits the transformer to the specified DataFrame and performs the transformation on the DataFrame.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            Tuple of updated data schema, fitted transformer and transformed DataFrame.
        """
        fitted_transformers: list[Any] = []
        for i, transformer in enumerate(self._transformers):
            logger.info(
                f"Executing composite transformation step {i+1}/{len(self._transformers)}: "
                f"{transformer.__class__.__name__}."
            )
            data_schema, fitted_transformer, dataframe = transformer.fit_transform(
                data_schema, dataframe, disable_cache=True
            )
            fitted_transformers.append(fitted_transformer)
            dataframe = dataframe.extract()
            logger.info(
                "Finished fitting and transforming composite transformation " f"step {i+1}/{len(self._transformers)}."
            )
        return data_schema, fitted_transformers, dataframe

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
