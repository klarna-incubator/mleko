"""Test suite for the `pipeline.steps.tune_step` module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.model.tune import BaseTuner
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.tune_step import TuneStep


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        feature1=[0.1, 0.2, 0.3],
        feature2=[2.0, 1.9, 1.8],
        target=[0, 1, 0],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example DataSchema."""
    return DataSchema(
        numerical=["feature1", "feature2"],
    )


class TestTuneStep:
    """Test suite for `pipeline.steps.tune_step.TuneStep`."""

    def test_init(self):
        """Should init the TuneStep with a tuner."""
        tuner = MagicMock(spec=BaseTuner)
        tune_step = TuneStep(
            tuner=tuner,
            inputs={"data_schema": "data_schema", "dataframe": "dataframe"},
            outputs={"hyperparameters": "hyperparameters", "score": "best_score", "metadata": "metadata"},
        )

        assert tune_step._tuner == tuner

    def test_execute(self, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame):
        """Should execute the tuning."""
        data_container = DataContainer(
            data={
                "data_schema": example_data_schema,
                "dataframe": example_vaex_dataframe,
            }
        )

        tuner = MagicMock(spec=BaseTuner)
        tuner.tune = MagicMock(return_value=({}, 0.0, None))

        tune_step = TuneStep(
            tuner=tuner,
            inputs={"data_schema": "data_schema", "dataframe": "dataframe"},
            outputs={"hyperparameters": "hyperparameters", "score": "best_score", "metadata": "metadata"},
            cache_group=None,
        )
        result = tune_step.execute(data_container, force_recompute=False, disable_cache=False)

        assert isinstance(result, DataContainer)
        assert result.data["hyperparameters"] == {}
        assert result.data["best_score"] == 0.0
        assert result.data["metadata"] is None

        tuner.tune.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["dataframe"], None, False, False
        )

    def test_wrong_data_type(self):
        """Should throw ValueError if not receiving a data schema or vaex DataFrame."""
        tuner = MagicMock(spec=BaseTuner)
        with pytest.raises(ValueError):
            TuneStep(
                tuner=tuner,
                inputs={"file_paths": "file_paths", "dataframe": "dataframe"},  # type: ignore
                outputs={"hyperparameters": "hyperparameters", "score": "best_score", "metadata": "metadata"},
            )

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        tuner = MagicMock(spec=BaseTuner)
        with pytest.raises(ValueError):
            TuneStep(tuner=tuner, inputs={}, outputs={"hyperparameters": "hyperparameters"})  # type: ignore

        with pytest.raises(ValueError):
            TuneStep(tuner=tuner, inputs={"dataframe": "df_clean"}, outputs={})  # type: ignore
