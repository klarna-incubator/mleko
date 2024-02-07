"""Module containing the ModelStep class."""

from __future__ import annotations

from typing import Optional, Union, cast

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import BaseModel, HyperparametersType
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import FitTransformAction, FitTransformPipelineStep
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr


logger = CustomLogger()
"""The logger for the module."""


class ModelStepInputFitType(TypedDict):
    """The input type of the ModelStep when action is 'fit'."""

    data_schema: Union[str, DataSchema]
    """DataSchema or the key for the DataSchema to be used for training."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be used for training."""

    validation_dataframe: Optional[Union[str, DataFrame]]
    """DataFrame or the key for the validation DataFrame to be used for training."""

    hyperparameters: Optional[Union[str, HyperparametersType]]
    """Hyperparameters or the key for the hyperparameters to be used for training."""


class ModelStepInputTransformType(TypedDict):
    """The input type of the ModelStep when action is 'transform'."""

    data_schema: Union[str, DataSchema]
    """DataSchema or the key for the DataSchema to be used for prediction."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be used for prediction."""


class ModelStepOutputFitType(TypedDict):
    """The output type of the ModelStep when action is 'fit'."""

    model: str
    """The key for the fitted model after training."""

    metrics: str
    """The key for the metrics dictionary after training."""


class ModelStepOutputTransformType(TypedDict):
    """The output type of the ModelStep when action is 'transform'."""

    dataframe: str
    """The key for the transformed DataFrame after prediction."""


class ModelStepOutputFitTransformType(ModelStepOutputFitType, ModelStepOutputTransformType):
    """The output type of the ModelStep when action is 'fit_transform'."""

    validation_dataframe: Optional[str]
    """The key for the transformed validation DataFrame after prediction."""


class ModelStep(FitTransformPipelineStep):
    """Pipeline step for model training and prediction."""

    _inputs: ModelStepInputFitType | ModelStepInputTransformType
    _outputs: ModelStepOutputFitType | ModelStepOutputTransformType | ModelStepOutputFitTransformType

    @auto_repr
    def __init__(
        self,
        model: BaseModel,
        action: FitTransformAction,
        inputs: ModelStepInputFitType | ModelStepInputTransformType,
        outputs: ModelStepOutputFitType | ModelStepOutputTransformType | ModelStepOutputFitTransformType,
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
            inputs: A dictionary of input keys following the `ModelStepInputFitType` or `ModelStepInputTransformType`
                schema, depending on the action.
            outputs: A list of output keys following the `ModelStepOutputFitType`, `ModelStepOutputTransformType`, or
                `ModelStepOutputFitTransformType` schema, depending on the action.
            cache_group: The cache group to use.
        """
        super().__init__(action, inputs, outputs, cache_group)
        self._model = model

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform actions using the configured model.

        Args:
            data_container: Contains the input.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the output of the action performed by the step, either the fitted model,
            the predictions on the DataFrame, or both.
        """
        data_schema = self._validate_and_get_input(self._inputs["data_schema"], DataSchema, data_container)
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        df_validation = hyperparameters = None
        if self._action != "transform":
            self._inputs = cast(ModelStepInputFitType, self._inputs)
            df_validation = self._validate_and_get_input(
                self._inputs["validation_dataframe"], DataFrame, data_container, is_optional=True
            )
            hyperparameters: HyperparametersType | None = self._validate_and_get_input(
                self._inputs["hyperparameters"], dict, data_container, is_optional=True
            )

        if self._action == "fit":
            self._outputs = cast(ModelStepOutputFitType, self._outputs)
            model, metrics = self._model.fit(
                data_schema, dataframe, df_validation, hyperparameters, self._cache_group, force_recompute
            )
            data_container.data[self._outputs["model"]] = model
            data_container.data[self._outputs["metrics"]] = metrics
        elif self._action == "transform":
            self._outputs = cast(ModelStepOutputTransformType, self._outputs)
            df = self._model.transform(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs["dataframe"]] = df
        elif self._action == "fit_transform":
            self._outputs = cast(ModelStepOutputFitTransformType, self._outputs)
            model, metrics, df, df_validation = self._model.fit_transform(
                data_schema, dataframe, df_validation, hyperparameters, self._cache_group, force_recompute
            )
            data_container.data[self._outputs["model"]] = model
            data_container.data[self._outputs["metrics"]] = metrics
            data_container.data[self._outputs["dataframe"]] = df
            if self._outputs["validation_dataframe"] is not None:
                data_container.data[self._outputs["validation_dataframe"]] = df_validation
        return data_container

    def _get_input_model(self) -> type[ModelStepInputFitType | ModelStepInputTransformType]:
        """Get the input type for the TransformStep.

        Returns:
            Input type for the TransformStep.
        """
        if self._action == "fit" or self._action == "fit_transform":
            return ModelStepInputFitType
        return ModelStepInputTransformType

    def _get_output_model(
        self,
    ) -> type[ModelStepOutputFitType | ModelStepOutputTransformType | ModelStepOutputFitTransformType]:
        """Get the output type for the TransformStep.

        Returns:
            Output type for the TransformStep.
        """
        if self._action == "fit":
            return ModelStepOutputFitType
        if self._action == "transform":
            return ModelStepOutputTransformType
        return ModelStepOutputFitTransformType
