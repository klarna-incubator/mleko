"""Module containing the TuneStep class."""

from __future__ import annotations

from typing import Union

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.model.tune.base_tuner import BaseTuner
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class TuneStepInputType(TypedDict):
    """The input type of the TuneStep."""

    data_schema: Union[str, DataSchema]
    """DataSchema or the key for the DataSchema to be used for hyperparameter tuning."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be used for hyperparameter tuning."""


class TuneStepOutputType(TypedDict):
    """The output type of the TuneStep."""

    hyperparameters: str
    """The key for the best hyperparameters after tuning."""

    score: str
    """The key for the best objective score after tuning."""

    metadata: str
    """The key for the optional metadata object after tuning."""


class TuneStep(PipelineStep):
    """Pipeline step for hyperparameter tuning."""

    _inputs: TuneStepInputType
    _outputs: TuneStepOutputType

    @auto_repr
    def __init__(
        self,
        tuner: BaseTuner,
        inputs: TuneStepInputType,
        outputs: TuneStepOutputType,
        cache_group: str | None = None,
    ) -> None:
        """Initialize the TuneStep with the specified tuner.

        Args:
            tuner: The tuner used for hyperparameter tuning.
            inputs: A dictionary of input keys following the `TuneStepInputType` schema.
            outputs: A dictionary of output keys following the `TuneStepOutputType` schema.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._tuner = tuner

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform hyperparameter tuning.

        Args:
            data_container: Contains the input.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the output of the tuning step. It contains the best hyperparameters, the best
            objective score, and an optional metadata object.
        """
        data_schema = self._validate_and_get_input(self._inputs["data_schema"], DataSchema, data_container)
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        hyperparameters, score, metadata = self._tuner.tune(data_schema, dataframe, self._cache_group, force_recompute)
        data_container.data[self._outputs["hyperparameters"]] = hyperparameters
        data_container.data[self._outputs["score"]] = score
        data_container.data[self._outputs["metadata"]] = metadata
        return data_container

    def _get_input_model(self) -> type[TuneStepInputType]:
        """Get the input model for the TuneStep.

        Returns:
            The input model for the TuneStep.
        """
        return TuneStepInputType

    def _get_output_model(self) -> type[TuneStepOutputType]:
        """Get the output model for the TuneStep.

        Returns:
            The output model for the TuneStep.
        """
        return TuneStepOutputType
