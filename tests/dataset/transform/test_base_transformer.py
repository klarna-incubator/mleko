"""Test suite for `dataset.transform.base_transformer`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.transform.base_transformer import BaseTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["1", "1", "0", "0"],
        b=["1", "1", "1", "1"],
        c=[None, "1", "1", "1"],
    )


class TestBaseTransformer:
    """Test suite for `dataset.transform.base_transformer.BaseTransformer`."""

    class DerivedTransformer(BaseTransformer):
        """Test class."""

        def _fit(self, _dataframe):
            """Fit transformer."""
            return 1337

        def _transform(self, _dataframe):
            """Transform Features."""
            return _dataframe

        def _fingerprint(self):
            """Return fingerprint."""
            return "fingerprint"

    def test_fit_and_transform(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should fit and transform dataframe."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        transformer = test_derived_transformer.fit(example_vaex_dataframe)
        df = test_derived_transformer.transform(example_vaex_dataframe)
        assert transformer == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

        with patch.object(self.DerivedTransformer, "_fit") as mocked_fit:
            test_derived_transformer.fit(example_vaex_dataframe)
            mocked_fit.assert_not_called()

        with patch.object(self.DerivedTransformer, "_transform") as mocked_transform:
            test_derived_transformer.transform(example_vaex_dataframe)
            mocked_transform.assert_not_called()

    def test_fit_transform(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should return vaex dataframe from feature_select method."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        transformer, df = test_derived_transformer.fit_transform(example_vaex_dataframe)
        assert transformer == 1337
        assert df.shape == (4, 3)
        assert df.column_names == ["a", "b", "c"]

    def test_error_on_transform_before_fit(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should raise error when transform is called before fit."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        with pytest.raises(RuntimeError):
            test_derived_transformer.transform(example_vaex_dataframe)
