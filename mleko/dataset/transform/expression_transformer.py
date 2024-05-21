"""Module for the expression transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import vaex
from typing_extensions import TypedDict

from mleko.cache.fingerprinters.json_fingerprinter import JsonFingerprinter
from mleko.dataset.data_schema import DataSchema, DataType
from mleko.utils import CustomLogger, auto_repr, get_column

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class ExpressionTransformerConfig(TypedDict):
    """A type alias for the configuration of the expression transformer."""

    expression: str
    """The `vaex` expression used to create the new feature."""

    type: DataType
    """The data type of the new feature."""

    is_meta: bool
    """A boolean indicating if the new feature is a metadata feature."""


class ExpressionTransformer(BaseTransformer):
    """Creates new features using `vaex` expressions."""

    @auto_repr
    def __init__(
        self,
        expressions: dict[str, ExpressionTransformerConfig],
        cache_directory: str | Path = "data/expression-transformer",
        cache_size: int = 1,
    ) -> None:
        """Initializes the transformer with the specified expressions.

        The expressions are a dictionary where the key is the name of the new feature and the value is a tuple
        containing the expression, the data type and a boolean indicating if the feaature is a metadata feature.
        The expression must be a valid `vaex` expression that can be evaluated on the DataFrame.

        Note:
            To translate a `vaex` vectorized statement to a valid `vaex` expression, use the `.expression` attribute.
            For example, the expression of `df["a"] + df["b"]` can be extracted using `(df["a"] + df["b"]).expression`.

        Args:
            expressions: A dictionary where the key is the name of the new feature and the value is a dictionary
                containing the expression, the data type and a boolean indicating if the feaature is a metadata feature.
                The expression must be a valid `vaex` expression that can be evaluated on the DataFrame.
            cache_directory: The directory where the cache will be stored locally.
            cache_size: The maximum number of cache entries to keep in the cache.

        Examples:
            >>> from mleko.dataset.data_schema import DataSchema
            >>> from mleko.dataset.transform import ExpressionTransformer
            >>> transformer = ExpressionTransformer(
            ...     expressions={
            ...         "sum": {"expression": "a + b", "type": "numerical", "is_meta": False},
            ...         "product": {"expression": "a * b", "type": "numerical", "is_meta": False},
            ...         "both_positive": {"expression": "(a > 0) & (b > 0)", "type": "boolean", "is_meta": True},
            ...     }
            ... )
            >>> df = vaex.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> ds = DataSchema(numerical=["a", "b"])
            >>> data_schema, _, transformed_df = transformer.fit_transform(ds, df)
            >>> transformed_df
            #    a    b    sum    product   both_positive
            0    1    4      5          4            True
            1    2    5      7         10            True
            2    3    6      9         18            True
            >>> data_schema # The 'both_positive' feature is a metadata feature and is not included in the data schema.
            DataSchema(numerical=['a', 'b', 'sum', 'product'])
        """
        super().__init__([], cache_directory, cache_size)
        self._transformer = expressions

    def _fit(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, dict[str, ExpressionTransformerConfig]]:
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
        for feature, config in self._transformer.items():
            logger.info(
                f"Creating new {config['type']!r} feature {feature!r} using expression {config['expression']!r}."
            )
            df[feature] = get_column(df, config["expression"]).as_arrow()
            if not config["is_meta"]:
                ds.add_feature(feature, config["type"])
        return ds, df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return super()._fingerprint(), JsonFingerprinter().fingerprint(self._transformer)
