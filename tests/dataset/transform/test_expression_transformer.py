"""Test suite for `dataset.transform.test_expression_transformer`."""

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.expression_transformer import ExpressionTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[1, 1, 0, 0],
        b=[3, 2, 7, 1],
        c=[5, -1, -1, 1],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(categorical=["a", "b", "c"])


class TestExpressionTransformer:
    """Test suite for `dataset.transform.expression_transformer.ExpressionTransformer`."""

    def test_expression_transformer(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly frequency encode the specified features."""
        expression_transformer = ExpressionTransformer(
            {
                "sum": ("astype(a + b + c, 'int32')", "numerical", False),
                "product": ("astype(a * b * c, 'int32')", "numerical", False),
                "all_positive": ("(a >= 0) & (b >= 0) & (c >= 0)", "boolean", True),
            },
            cache_directory=temporary_directory,
        )

        ds, _, df = expression_transformer._fit_transform(example_data_schema, example_vaex_dataframe)

        assert df["sum"].tolist() == [9, 2, 6, 2]  # type: ignore
        assert df["product"].tolist() == [15, -2, 0, 0]  # type: ignore
        assert df["all_positive"].tolist() == [True, False, False, True]  # type: ignore
        assert ds.get_type("sum") == "numerical"
        assert ds.get_type("product") == "numerical"
        assert "all_positive" not in ds.get_features()

    def test_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly frequency encode features and use cache if possible."""
        ExpressionTransformer(
            {
                "sum": ("astype(a + b + where(isna(c), 0, c), 'int32')", "numerical", False),
                "product": ("astype(a * b * where(isna(c), 1, c), 'int32')", "numerical", False),
            },
            cache_directory=temporary_directory,
        ).fit_transform(example_data_schema, example_vaex_dataframe)

        with patch.object(ExpressionTransformer, "_fit_transform") as mocked_fit_transform:
            ExpressionTransformer(
                {
                    "sum": ("astype(a + b + where(isna(c), 0, c), 'int32')", "numerical", False),
                    "product": ("astype(a * b * where(isna(c), 1, c), 'int32')", "numerical", False),
                },
                cache_directory=temporary_directory,
            ).fit_transform(example_data_schema, example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()
