"""Module containing the FeatureSelectStep class.""" ""
from __future__ import annotations

from typing import Union, cast

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import FitTransformAction, FitTransformPipelineStep
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr


logger = CustomLogger()
"""The logger for the module."""


class FeatureSelectStepInputType(TypedDict):
    """The input type of the FeatureSelectStep."""

    data_schema: Union[str, DataSchema]
    """DataSchema or the key for the DataSchema to be used for feature selection."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be used for feature selection."""


class FeatureSelectStepOutputFitType(TypedDict):
    """The output type of the FeatureSelectStep when action is 'fit'."""

    data_schema: str
    """The key for the updated DataSchema after feature selection."""

    feature_selector: str
    """The key for the fitted feature selector after feature selection."""


class FeatureSelectStepOutputTransformType(TypedDict):
    """The output type of the FeatureSelectStep when action is 'transform'."""

    data_schema: str
    """The key for the updated DataSchema after feature selection."""

    dataframe: str
    """The key for the transformed DataFrame after feature selection."""


class FeatureSelectStepOutputFitTransformType(FeatureSelectStepOutputFitType, FeatureSelectStepOutputTransformType):
    """The output type of the FeatureSelectStep when action is 'fit_transform'."""

    pass


class FeatureSelectStep(FitTransformPipelineStep):
    """Pipeline step that selects a subset of features from a DataFrame."""

    _inputs: FeatureSelectStepInputType
    _outputs: (
        FeatureSelectStepOutputFitType | FeatureSelectStepOutputTransformType | FeatureSelectStepOutputFitTransformType
    )

    @auto_repr
    def __init__(
        self,
        feature_selector: BaseFeatureSelector,
        action: FitTransformAction,
        inputs: FeatureSelectStepInputType,
        outputs: (
            FeatureSelectStepOutputFitType
            | FeatureSelectStepOutputTransformType
            | FeatureSelectStepOutputFitTransformType
        ),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the FeatureSelectStep with the specified feature selector.

        Args:
            feature_selector: The FeatureSelector responsible for handling feature selection.
            action: The action to perform, one of "fit", "transform", or "fit_transform".
            inputs: A dictionary of input keys following the `FeatureSelectStepInputType` schema.
            outputs: A dictionary of output keys following the `FeatureSelectStepOutputFitType`,
                `FeatureSelectStepOutputTransformType`, or `FeatureSelectStepOutputFitTransformType` schema depending
                on the action.
            cache_group: The cache group to use.
        """
        super().__init__(action, inputs, outputs, cache_group)
        self._feature_selector = feature_selector

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform feature selection using the configured feature selector.

        Args:
            data_container: Contains the input DataFrame.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the result depending on the action.
        """
        data_schema = self._validate_and_get_input(self._inputs["data_schema"], DataSchema, data_container)
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        if self._action == "fit":
            self._outputs = cast(FeatureSelectStepOutputFitType, self._outputs)
            ds, feature_selector = self._feature_selector.fit(
                data_schema, dataframe, self._cache_group, force_recompute
            )
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["feature_selector"]] = feature_selector
        elif self._action == "transform":
            self._outputs = cast(FeatureSelectStepOutputTransformType, self._outputs)
            ds, df = self._feature_selector.transform(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["dataframe"]] = df
        elif self._action == "fit_transform":
            self._outputs = cast(FeatureSelectStepOutputFitTransformType, self._outputs)
            ds, feature_selector, df = self._feature_selector.fit_transform(
                data_schema, dataframe, self._cache_group, force_recompute
            )
            data_container.data[self._outputs["data_schema"]] = ds
            data_container.data[self._outputs["feature_selector"]] = feature_selector
            data_container.data[self._outputs["dataframe"]] = df
        return data_container

    def _get_input_model(self) -> type[FeatureSelectStepInputType]:
        """Get the input type for the FeatureSelectStep.

        Returns:
            Input type for the FeatureSelectStep.
        """
        return FeatureSelectStepInputType

    def _get_output_model(
        self,
    ) -> type[
        FeatureSelectStepOutputFitType | FeatureSelectStepOutputTransformType | FeatureSelectStepOutputFitTransformType
    ]:
        """Get the output type for the FeatureSelectStep.

        Returns:
            Output type for the FeatureSelectStep.
        """
        if self._action == "fit":
            return FeatureSelectStepOutputFitType
        if self._action == "transform":
            return FeatureSelectStepOutputTransformType
        return FeatureSelectStepOutputFitTransformType
