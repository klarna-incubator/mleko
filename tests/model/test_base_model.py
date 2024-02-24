"""Test suite for `model.base_model`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import BaseModel


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
    """Return an example DataSchema."""
    return DataSchema(
        categorical=["a", "b", "c"],
    )


class TestBaseModel:
    """Test suite for `model.base_model.BaseModel`."""

    class DerivedModel(BaseModel):
        """Test class."""

        def _fit(self, _data_schema, _train_dataframe, _validation_dataframe, _hyperparameters):
            """Fit transformer."""
            return 1337, {}

        def _transform(self, _data_schema, _dataframe):
            """Transform Features."""
            return _dataframe

        def _fingerprint(self):
            """Return fingerprint."""
            return "fingerprint"

        def _default_features(self, _data_schema):
            """Return default features."""
            return ()

    def test_fit_and_transform(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should fit and transform dataframe."""
        test_derived_model = self.DerivedModel(None, None, temporary_directory, 1)

        model, _ = test_derived_model.fit(example_data_schema, example_vaex_dataframe, example_vaex_dataframe, {})
        df = test_derived_model.transform(example_data_schema, example_vaex_dataframe)
        assert model == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

        with patch.object(self.DerivedModel, "_fit") as mocked_fit:
            test_derived_model.fit(example_data_schema, example_vaex_dataframe, example_vaex_dataframe, {})
            mocked_fit.assert_not_called()

        with patch.object(self.DerivedModel, "_transform") as mocked_transform:
            test_derived_model.transform(example_data_schema, example_vaex_dataframe)
            mocked_transform.assert_not_called()

    def test_fit_transform(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should return vaex dataframe from feature_select method."""
        test_derived_model = self.DerivedModel(None, None, temporary_directory, 1)

        model, _, df, _ = test_derived_model.fit_transform(
            example_data_schema, example_vaex_dataframe, example_vaex_dataframe, {}
        )
        assert model == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

    def test_error_on_transform_before_fit(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should raise error when transform is called before fit."""
        test_derived_model = self.DerivedModel(None, None, temporary_directory, 1)

        with pytest.raises(RuntimeError):
            test_derived_model.transform(example_data_schema, example_vaex_dataframe)

    def test_error_on_mutually_exclusive_arguments(self, temporary_directory: Path):
        """Should raise error when both `features` and `ignore_features` are passed."""
        with pytest.raises(ValueError):
            self.DerivedModel([], [], temporary_directory, 1)
