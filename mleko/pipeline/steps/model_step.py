"""Module containing the ModelStep class."""
from __future__ import annotations

from typing import Literal

from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import BaseModel
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class ModelStep(PipelineStep):
    """Pipeline step for model training and prediction."""

    _num_inputs = 4
    """Number of inputs expected by the ModelStep."""

    _num_outputs = 1
    """Number of outputs expected by the ModelStep."""

    @auto_repr
    def __init__(
        self,
        model: BaseModel,
        action: Literal["fit", "transform", "fit_transform"],
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the ModelStep with the specified model.

        The action parameter specifies whether the model should be fitted, transformed, or both. If the action
        is "fit" the `ModelStep` will return the fitted model. If the action is "transform" the
        `ModelStep` will return the predictions of the DataFrame. If the action is "fit_transform" the `ModelStep`
        will return both the fitted model and the predictions of the DataFrame in the specified order.

        Note:
            The hyperparameters input is optional. If it is not provided, the model will be trained using the
            default hyperparameters configured in the model.

        Args:
            model: The model used for training and prediction.
            action: The action to perform, one of "fit", "transform", or "fit_transform".
            inputs: List or tuple of input keys expected by this step. If the action is "fit" or "fit_transform",
                should contain four keys, corresponding to the DataSchema, the training DataFrame to be fitted,
                the validation DataFrame to be used for validation, and the hyperparameters to be used for training.
                If the action is "transform", should contain two keys, corresponding to the DataSchema, and the
                DataFrame to be transformed.
            outputs: List or tuple of output keys produced by this step. If the action is "fit", should contain two
                keys, corresponding to the fitted model and the metrics dictionary. If the action is "transform",
                should contain a single key, corresponding to the transformed DataFrame. If the action is
                "fit_transform", should contain four keys, corresponding to the fitted model, the metrics
                dictionary, the transformed training DataFrame, and the transformed validation DataFrame.
            cache_group: The cache group to use.

        Raises:
            ValueError: If action is not one of "fit", "transform", or "fit_transform".
        """
        if action not in ("fit", "transform", "fit_transform"):
            raise ValueError(
                f"Invalid action: {action}. Expected one of 'fit', 'transform', or 'fit_transform'."
            )  # pragma: no cover

        if action == "fit":
            self._num_outputs = 2
        if action == "transform":
            self._num_inputs = 2
        if action == "fit_transform":
            self._num_outputs = 4

        super().__init__(inputs, outputs, cache_group)
        self._model = model
        self._action = action

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform actions using the configured model.

        Args:
            data_container: Contains the input.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the output of the action performed by the step, either the fitted model,
            the predictions on the DataFrame, or both.
        """
        data_schema = data_container.data[self._inputs[0]]
        if not isinstance(data_schema, DataSchema):
            raise ValueError(f"Invalid data type: {type(data_schema)}. Expected DataSchema.")

        dataframe = data_container.data[self._inputs[1]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"Invalid data type: {type(dataframe)}. Expected vaex DataFrame.")

        df_validation = hyperparameters = None
        if self._action != "transform":
            df_validation = data_container.data[self._inputs[2]]
            if not isinstance(df_validation, DataFrame):
                raise ValueError(f"Invalid data type: {type(df_validation)}. Expected vaex DataFrame.")

            hyperparameters = {}
            if len(self._inputs) == 4:
                hyperparameters = data_container.data.get(self._inputs[3])
            if not isinstance(hyperparameters, dict):
                raise ValueError(f"Invalid data type: {type(hyperparameters)}. Expected HyperparametersType.")

        if self._action == "fit" and df_validation is not None:
            model, metrics = self._model.fit(
                data_schema, dataframe, df_validation, hyperparameters, self._cache_group, force_recompute
            )
            data_container.data[self._outputs[0]] = model
            data_container.data[self._outputs[1]] = metrics
        elif self._action == "transform":
            df = self._model.transform(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = df
        elif self._action == "fit_transform" and df_validation is not None:
            model, metrics, df, df_validation = self._model.fit_transform(
                data_schema, dataframe, df_validation, hyperparameters, self._cache_group, force_recompute
            )
            data_container.data[self._outputs[0]] = model
            data_container.data[self._outputs[1]] = metrics
            data_container.data[self._outputs[2]] = df
            data_container.data[self._outputs[3]] = df_validation
        return data_container

    def _validate_inputs(self) -> None:
        """Check the input keys for compliance with this step's requirements.

        Raises:
            ValueError: If the PipelineStep has an invalid number of inputs.
        """
        if len(self._inputs) != self._num_inputs and len(self._inputs) != self._num_inputs - 1:
            raise ValueError(f"{self.__class__.__name__} must have exactly {self._num_inputs} input(s).")
