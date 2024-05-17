"""Module for the expression transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex

from mleko.cache.fingerprinters.json_fingerprinter import JsonFingerprinter
from mleko.dataset.data_schema import DataSchema, DataType
from mleko.utils import CustomLogger, auto_repr, get_column

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class ExpressionTransformer(BaseTransformer):
    """Creates new features using `vaex` expressions."""

    @auto_repr
    def __init__(
        self,
        expressions: dict[str, tuple[str, DataType]],
        cache_directory: str | Path = "data/expression-transformer",
        cache_size: int = 1,
    ) -> None:
        """Initializes the transformer with the specified expressions.

        The expressions are a dictionary where the key is the name of the new feature and the value is a tuple
        containing the expression and the data type of the new feature. The expression must be a valid `vaex`
        expression that can be evaluated on the DataFrame.

        Note:
            To translate a `vaex` vectorized statement to a valid `vaex` expression, use the `.expression` attribute.
            For example, the expression of `df["a"] + df["b"]` can be extracted using `(df["a"] + df["b"]).expression`.

        Args:
            expressions: A dictionary where the key is the name of the new feature and the value is a tuple containing
                the expression and the data type of the new feature.
            cache_directory: The directory where the cache will be stored locally.
            cache_size: The maximum number of cache entries to keep in the cache.

        Examples:
            >>> from mleko.dataset.transform import ExpressionTransformer
            >>> transformer = ExpressionTransformer(
            ...     expressions={
            ...         "sum": ("df['a'] + df['b']", "numerical"),
            ...         "product": ("df['a'] * df['b']", "numerical"),
            ...     }
            ... )
            >>> df = vaex.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> _, _, transformed_df = transformer.fit_transform(df)
            >>> transformed_df
            #    a    b    sum    product
            0    1    4      5          4
            1    2    5      7         10
            2    3    6      9         18
        """
        super().__init__([], cache_directory, cache_size)
        self._transformer = expressions

    def _fit(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, dict[str, tuple[str, DataType]]]:
        """No fitting is required for the expression transformer.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to fit.

        Returns:
            The data schema and the transformer.
        """
        return data_schema, self._transformer

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the specified features in the DataFrame using the expressions provided.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            The transformed data schema and the transformed DataFrame.
        """
        df = dataframe.copy()
        ds = data_schema.copy()
        for feature, (expression, data_type) in self._transformer.items():
            logger.info(f"Creating new {data_type!r} feature {feature!r} using expression {expression!r}.")
            df[feature] = get_column(df, expression).as_arrow()
            ds.add_feature(feature, data_type)
        print(df["a"])
        return ds, df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), JsonFingerprinter().fingerprint(self._transformer)
