"""Test suite for the `pipeline.steps.model_step` module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import BaseModel
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.model_step import ModelStep


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


class TestModelStep:
    """Test suite for `pipeline.steps.model_step.ModelStep`."""

    def test_init(self):
        """Should init the ModelStep with a model."""
        model = MagicMock(spec=BaseModel)
        model_step = ModelStep(
            model=model,
            action="fit",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "df_train",
                "validation_dataframe": "df_validate",
                "hyperparameters": None,
            },
            outputs={"model": "model", "metrics": "metrics"},
        )

        assert model_step._model == model

    def test_execute_fit_transform(self, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame):
        """Should execute the fitting and transforming."""
        data_container = DataContainer(
            data={
                "data_schema": example_data_schema,
                "df_train": example_vaex_dataframe,
                "df_validate": example_vaex_dataframe,
            }
        )

        model = MagicMock(spec=BaseModel)
        model.fit_transform = MagicMock(return_value=("model", "metrics", "df_train", "df_validate"))

        model_step = ModelStep(
            model=model,
            action="fit_transform",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "df_train",
                "validation_dataframe": "df_validate",
                "hyperparameters": None,
            },
            outputs={
                "model": "model",
                "metrics": "metrics",
                "dataframe": "df_train_out",
                "validation_dataframe": "df_validate_out",
            },
            cache_group=None,
        )
        result = model_step.execute(data_container, force_recompute=False, disable_cache=False)

        assert isinstance(result, DataContainer)
        assert result.data["model"] == "model"

        model.fit_transform.assert_called_once_with(
            data_container.data["data_schema"],
            data_container.data["df_train"],
            data_container.data["df_validate"],
            None,
            None,
            False,
            False,
        )

    def test_execute_fit_and_transform(self, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame):
        """Should execute the fit_transforming."""
        data_container = DataContainer(
            data={
                "data_schema": example_data_schema,
                "df_train": example_vaex_dataframe,
                "df_validate": example_vaex_dataframe,
                "hyperparameters": {"param1": "value1"},
            }
        )

        model = MagicMock(spec=BaseModel)
        model.fit = MagicMock(return_value=("model", "metrics"))
        model.transform = MagicMock(return_value="df")

        model_step_fit = ModelStep(
            model=model,
            action="fit",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "df_train",
                "validation_dataframe": "df_validate",
                "hyperparameters": "hyperparameters",
            },
            outputs={"model": "model", "metrics": "metrics"},
            cache_group=None,
        )
        model_step_fit_result = model_step_fit.execute(data_container, force_recompute=False, disable_cache=False)
        assert isinstance(model_step_fit_result, DataContainer)
        assert model_step_fit_result.data["model"] == "model"

        model_step_transform = ModelStep(
            model=model,
            action="transform",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "df_train",
            },
            outputs={"dataframe": "df"},
            cache_group=None,
        )
        model_step_transform_result = model_step_transform.execute(
            data_container, force_recompute=False, disable_cache=False
        )
        assert isinstance(model_step_transform_result, DataContainer)
        assert model_step_transform_result.data["df"] == "df"

        model.fit.assert_called_once_with(
            data_container.data["data_schema"],
            data_container.data["df_train"],
            data_container.data["df_validate"],
            data_container.data["hyperparameters"],
            None,
            False,
            False,
        )
        model.transform.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_train"], None, False, False
        )

    def test_wrong_data_type(self):
        """Should throw ValueError if not receiving correct inputs."""
        file_paths = [str]
        data_container = DataContainer(data={"data_schema": file_paths})  # type: ignore

        model = MagicMock(spec=BaseModel)
        model_step = ModelStep(
            model=model,
            action="fit",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "data_schema",
                "validation_dataframe": "data_schema",
                "hyperparameters": "data_schema",
            },
            outputs={"model": "model", "metrics": "metrics"},
        )

        with pytest.raises(ValueError):
            model_step.execute(data_container, force_recompute=False, disable_cache=False)

    def test_none_on_required_input(self, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame):
        """Should throw ValueError if not receiving correct inputs."""
        data_container = DataContainer(
            data={
                "data_schema": example_data_schema,
                "df_train": example_vaex_dataframe,
                "df_validate": example_vaex_dataframe,
                "hyperparameters": {"param1": "value1"},
            }
        )
        model = MagicMock(spec=BaseModel)
        model_step = ModelStep(
            model=model,
            action="fit",
            inputs={
                "data_schema": "data_schema",
                "dataframe": "df_train_wrong_name",
                "validation_dataframe": "df_validate",
                "hyperparameters": "hyperparameters",
            },
            outputs={"model": "model", "metrics": "metrics"},
        )

        with pytest.raises(ValueError):
            model_step.execute(data_container, force_recompute=False, disable_cache=False)

        model_step = ModelStep(
            model=model,
            action="fit",
            inputs={
                "data_schema": "data_schema",
                "dataframe": None,  # type: ignore
                "validation_dataframe": "df_validate",
                "hyperparameters": "hyperparameters",
            },
            outputs={"model": "model", "metrics": "metrics"},
        )

        with pytest.raises(ValueError):
            model_step.execute(data_container, force_recompute=False, disable_cache=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        model = MagicMock(spec=BaseModel)
        with pytest.raises(ValueError):
            ModelStep(model=model, action="fit", inputs={}, outputs={"dataframe": "df"})  # type: ignore

        with pytest.raises(ValueError):
            ModelStep(
                model=model,
                action="fit_transform",
                inputs={  # type: ignore
                    "data_schema": "data_schema",
                    "dataframe": "data_schema",
                    "validation_dataframe": "data_schema",
                    "hyperparameters": 5,
                },
                outputs={"dataframe": "df"},
            )
