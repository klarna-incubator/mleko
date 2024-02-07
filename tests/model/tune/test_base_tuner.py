"""Test suite for `model.tune.base_tuner`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.model.tune.base_tuner import BaseTuner


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


class TestBaseTuner:
    """Test suite for `model.base_tuner.BaseTuner`."""

    class DerivedTuner(BaseTuner):
        """Test class."""

        def _tune(self, _data_schema, _dataframe):
            """Hyperparameter tune."""
            return {}, 1337, {}

        def _fingerprint(self):
            """Return fingerprint."""
            return "fingerprint"

    def test_tune(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should tune."""
        test_derived_tuner = self.DerivedTuner(temporary_directory, 1)

        params, score, info = test_derived_tuner.tune(example_data_schema, example_vaex_dataframe)
        assert params == {}
        assert score == 1337
        assert info == {}

        with patch.object(self.DerivedTuner, "_tune") as mocked_tune:
            test_derived_tuner.tune(example_data_schema, example_vaex_dataframe)
            mocked_tune.assert_not_called()
