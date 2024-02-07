"""Module containing the TransformStep class."""

from __future__ import annotations

from typing import Union, cast

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.base_transformer import BaseTransformer
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import FitTransformAction, FitTransformPipelineStep
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr


logger = CustomLogger()
"""The logger for the module."""


class TransformStepInputType(TypedDict):
    """The input type of the TransformStep."""

    data_schema: Union[str, DataSchema]
    """DataSchema or the key for the DataSchema to be used for transformation."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be transformed."""


class TransformStepOutputFitType(TypedDict):
    """The output type of the TransformStep when action is 'fit'."""

    data_schema: str
    """The key for the updated DataSchema after transformation."""

    transformer: str
    """The key for the fitted transformer after transformation."""


class TransformStepOutputTransformType(TypedDict):
    """The output type of the TransformStep when action is 'transform'."""

    data_schema: str
    """The key for the updated DataSchema after transformation."""

    dataframe: str
    """The key for the transformed DataFrame after transformation."""


class TransformStepOutputFitTransformType(TransformStepOutputFitType, TransformStepOutputTransformType):
    """The output type of the TransformStep when action is 'fit_transform'."""

    pass


class TransformStep(FitTransformPipelineStep):
    """Pipeline step for transformation of features in DataFrame."""

    _inputs: TransformStepInputType
    _outputs: TransformStepOutputFitType | TransformStepOutputTransformType | TransformStepOutputFitTransformType

    @auto_repr
    def __init__(
        self,
        transformer: BaseTransformer,
        action: FitTransformAction,
        inputs: TransformStepInputType,
        outputs: TransformStepOutputFitType | TransformStepOutputTransformType | TransformStepOutputFitTransformType,
        cache_group: str | None = None,
    ) -> None:
        """Initialize the TransformStep with the specified transformer.

        Args:
            transformer: The Transformer responsible for handling feature transformation.
            action: The action to perform, one of "fit", "transform", or "fit_transform".
            inputs: A dictionary of input keys following the `TransformStepInputType` schema.
            outputs: A dictionary of output keys following one of `TransformStepOutputFitType`,
                `TransformStepOutputTransformType`, or `TransformStepOutputFitTransformType` schemas, depending on
                the action.
            cache_group: The cache group to use.
        """
        super().__init__(action, inputs, outputs, cache_group)
        self._transformer = transformer

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform transformation using the configured transformer.

        Args:
            data_container: Contains the input DataFrame.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the result depending on the action.
        """
        data_schema = self._validate_and_get_input(self._inputs["data_schema"], DataSchema, data_container)
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        if self._action == "fit":
            self._outputs = cast(TransformStepOutputFitType, self._outputs)
            ds, transformer = self._transformer.fit(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["transformer"]] = transformer
        elif self._action == "transform":
            self._outputs = cast(TransformStepOutputTransformType, self._outputs)
            ds, df = self._transformer.transform(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["dataframe"]] = df
        elif self._action == "fit_transform":
            self._outputs = cast(TransformStepOutputFitTransformType, self._outputs)
            ds, transformer, df = self._transformer.fit_transform(
                data_schema, dataframe, self._cache_group, force_recompute
            )
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["transformer"]] = transformer
            data_container.data[self._outputs["dataframe"]] = df
        return data_container

    def _get_input_model(self) -> type[TransformStepInputType]:
        """Get the input type for the TransformStep.

        Returns:
            Input type for the TransformStep.
        """
        return TransformStepInputType

    def _get_output_model(
        self,
    ) -> type[TransformStepOutputFitType | TransformStepOutputTransformType | TransformStepOutputFitTransformType]:
        """Get the output type for the TransformStep.

        Returns:
            Output type for the TransformStep.
        """
        if self._action == "fit":
            return TransformStepOutputFitType
        if self._action == "transform":
            return TransformStepOutputTransformType
        return TransformStepOutputFitTransformType
