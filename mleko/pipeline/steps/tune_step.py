"""Module containing the TuneStep class."""
from __future__ import annotations

from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.model.tune.base_tuner import BaseTuner
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class TuneStep(PipelineStep):
    """Pipeline step for hyperparameter tuning."""

    _num_inputs = 2
    """Number of inputs expected by the TuneStep."""

    _num_outputs = 3
    """Number of outputs expected by the TuneStep."""

    @auto_repr
    def __init__(
        self,
        tuner: BaseTuner,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the TuneStep with the specified tuner.

        Args:
            tuner: The tuner used for hyperparameter tuning.
            inputs: List or tuple of input keys expected by this step. Should contain two keys, corresponding to the
                DataSchema and the DataFrame to be used for hyperparameter tuning. The two inputs are passed to the
                tunes's objective function.
            outputs: List or tuple of output keys produced by this step. Should contain three keys, corresponding to
                the best hyperparameters, the best objective score, and an optional metadata object.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._tuner = tuner

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform hyperparameter tuning.

        Args:
            data_container: Contains the input.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the output of the tuning step. It contains the best hyperparameters, the best
            objective score, and an optional metadata object.
        """
        data_schema = data_container.data[self._inputs[0]]
        if not isinstance(data_schema, DataSchema):
            raise ValueError(f"Invalid data type: {type(data_schema)}. Expected DataSchema.")

        dataframe = data_container.data[self._inputs[1]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"Invalid data type: {type(dataframe)}. Expected vaex DataFrame.")

        hyperparameters, score, metadata = self._tuner.tune(data_schema, dataframe, self._cache_group, force_recompute)
        data_container.data[self._outputs[0]] = hyperparameters
        data_container.data[self._outputs[1]] = score
        data_container.data[self._outputs[2]] = metadata
        return data_container
