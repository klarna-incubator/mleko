"""Test suite for `model.tune.optuna_tuner`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import optuna
import pytest
import vaex
from optuna.samplers import (
    BruteForceSampler,
    CmaEsSampler,
    GridSampler,
    NSGAIIISampler,
    NSGAIISampler,
    PartialFixedSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from mleko.dataset.data_schema import DataSchema
from mleko.model.tune.optuna_tuner import OptunaTuner


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=[1, 1, 0, 0],
        b=[1, 1, 1, 1],
        c=[None, 1, 1, 1],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example DataSchema."""
    return DataSchema(
        numerical=["a", "b", "c"],
    )


class TestOptunaTuner:
    """Test suite for `model.optuna_tuner.OptunaTuner`."""

    def test_tune_single_objective(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should run hyperparameter tuning with `Optuna` towards a single objective."""

        def objective_function(trial, _data_schema, _dataframe):
            hyperparameters = {
                "x": trial.suggest_float("x", 1e-8, 10.0, log=True),
            }

            return len(example_vaex_dataframe["a"].tolist()) * hyperparameters["x"]  # type: ignore

        test_tuner = OptunaTuner(
            cache_directory=temporary_directory,
            objective_function=objective_function,
            direction="maximize",
            num_trials=10,
            random_state=42,
        )

        params, score, study = test_tuner.tune(example_data_schema, example_vaex_dataframe)
        assert params == {"x": 3.6010467344475403}
        assert score == 14.404186937790161
        assert isinstance(study, optuna.study.Study)

        with patch.object(OptunaTuner, "_tune") as mocked_tune:
            mocked_tune.return_value = {}, 1337, {}
            test_tuner.tune(example_data_schema, example_vaex_dataframe)
            mocked_tune.assert_not_called()

    def test_tune_multi_objective(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should run hyperparameter tuning with `Optuna` towards a multi objective."""

        def objective_function(trial, _data_schema, _dataframe):
            hyperparameters = {
                "x": trial.suggest_float("x", 1e-8, 10.0, log=True),
                "y": trial.suggest_float("y", 0, 10.0),
            }

            return (
                example_vaex_dataframe["a"].sum() * hyperparameters["x"],  # type: ignore
                example_vaex_dataframe["b"].sum() * hyperparameters["x"] ** hyperparameters["y"],  # type: ignore
            )

        test_tuner = OptunaTuner(
            cache_directory=temporary_directory,
            objective_function=objective_function,
            direction=["maximize", "maximize"],
            num_trials=10,
            random_state=42,
        )

        params, score, study = test_tuner.tune(example_data_schema, example_vaex_dataframe)
        assert params == {"x": 0.31044435499483225, "y": 2.1233911067827616}
        assert score == [0.6208887099896645, 0.3336897299758702]
        assert isinstance(study, optuna.study.Study)

    @pytest.mark.parametrize(
        "sampler",
        [
            (sampler)
            for sampler in [
                BruteForceSampler(),
                CmaEsSampler(),
                GridSampler(search_space={"x": [1, 2, 3]}),
                NSGAIIISampler(),
                NSGAIISampler(),
                PartialFixedSampler(fixed_params={"x": 1}, base_sampler=RandomSampler()),
                QMCSampler(),
                RandomSampler(),
                TPESampler(),
            ]
        ],
    )
    def test_tune_all_samplers(
        self,
        sampler,
        temporary_directory: Path,
    ):
        """Should successfully lock random states for all samplers."""

        def objective_function(trial, _data_schema, _dataframe):
            return 1

        _ = OptunaTuner(
            cache_directory=temporary_directory,
            objective_function=objective_function,
            direction=["maximize", "maximize"],
            sampler=sampler,
            num_trials=10,
            random_state=42,
        )

    def test_tune_with_storage(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame, example_data_schema: DataSchema
    ):
        """Should run hyperparameter tuning with `Optuna` towards a single objective and use a storage."""

        def objective_function(trial, _data_schema, _dataframe):
            hyperparameters = {
                "x": trial.suggest_float("x", 1e-8, 10.0, log=True),
            }

            return len(example_vaex_dataframe["a"].tolist()) * hyperparameters["x"]  # type: ignore

        test_tuner = OptunaTuner(
            cache_directory=temporary_directory,
            objective_function=objective_function,
            direction="maximize",
            storage_name="test",
            num_trials=10,
            random_state=42,
        )

        params, score, study = test_tuner._tune(example_data_schema, example_vaex_dataframe)
        assert params == {"x": 3.6010467344475403}
        assert score == 14.404186937790161
        assert isinstance(study, optuna.study.Study)
