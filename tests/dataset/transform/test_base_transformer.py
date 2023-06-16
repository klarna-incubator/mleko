"""Test suite for `dataset.transform.base_transformer`."""
from __future__ import annotations

from pathlib import Path

import vaex

from mleko.dataset.transform.base_transformer import BaseTransformer


class TestBaseTransformer:
    """Test suite for `dataset.transform.base_transformer.BaseTransformer`."""

    class DerivedTransformer(BaseTransformer):
        """Test class."""

        def transform(self, _dataframe):
            """Transform Features."""
            return self._transform(_dataframe)

        def _transform(self, dataframe):
            """Transform Features."""
            return vaex.from_arrays(a=[1, 2, 3], b=[4, 5, 6])

        def _fingerprint(self, dataframe):
            """Return fingerprint."""
            return "fingerprint"

    def test_abstract_methods(self, temporary_directory: Path):
        """Should return vaex dataframe from feature_select method."""
        test_derived_transformer = self.DerivedTransformer(temporary_directory, [], 1)

        df_train = test_derived_transformer.transform([])
        assert df_train.shape == (3, 2)
        assert df_train.column_names == ["a", "b"]
