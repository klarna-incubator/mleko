"""Test suite for `dataset.transform.base_transformer`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.base_transformer import BaseTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["1", "1", "0", "0"],
        b=["1", "1", "1", "1"],
        c=[None, "1", "1", "1"],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(categorical=["a", "b", "c"])


class TestBaseTransformer:
    """Test suite for `dataset.transform.base_transformer.BaseTransformer`."""

    class DerivedTransformer(BaseTransformer):
        """Test class."""

        def _fit(self, data_schema, _dataframe):
            """Fit transformer."""
            return data_schema, 1337

        def _transform(self, data_schema, _dataframe):
            """Transform Features."""
            return data_schema, _dataframe

        def _fingerprint(self):
            """Return fingerprint."""
            return "fingerprint"

    def test_fit_and_transform(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should fit and transform dataframe."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        _, transformer = test_derived_transformer.fit(example_data_schema, example_vaex_dataframe)
        _, df = test_derived_transformer.transform(example_data_schema, example_vaex_dataframe)
        assert transformer == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

        with patch.object(self.DerivedTransformer, "_fit") as mocked_fit:
            test_derived_transformer.fit(example_data_schema, example_vaex_dataframe)
            mocked_fit.assert_not_called()

        with patch.object(self.DerivedTransformer, "_transform") as mocked_transform:
            test_derived_transformer.transform(example_data_schema, example_vaex_dataframe)
            mocked_transform.assert_not_called()

    def test_fit_transform(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe from feature_select method."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        _, transformer, df = test_derived_transformer.fit_transform(example_data_schema, example_vaex_dataframe)
        assert transformer == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

    def test_error_on_transform_before_fit(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should raise error when transform is called before fit."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        with pytest.raises(RuntimeError):
            test_derived_transformer.transform(example_data_schema, example_vaex_dataframe)
