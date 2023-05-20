"""Module handling dataframe splitting.

This module contains the SplitStep class, which is responsible for splitting a DataFrame into two parts.
It uses a BaseDataSplitter to perform the actual splitting.
"""
from __future__ import annotations

from vaex import DataFrame

from mleko.data.splitters import BaseDataSplitter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class SplitStep(PipelineStep):
    """Pipeline step that splits a DataFrame in two."""

    _num_inputs = 1
    """Number of inputs expected by the SplitStep."""

    _num_outputs = 2
    """Number of outputs expected by the SplitStep."""

    @auto_repr
    def __init__(
        self,
        splitter: BaseDataSplitter,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize the SplitStep with the specified data splitter.

        Args:
            splitter: The DataSplitter responsible for handling data splitting.
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
        """
        super().__init__(inputs, outputs)
        self._splitter = splitter

    def execute(self, data_container: DataContainer) -> DataContainer:
        """Perform data splitting using the configured splitter.

        Args:
            data_container: Contains the DataFrame to be split.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the split data as two vaex DataFrames.
        """
        dataframe = data_container.data[self.inputs[0]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError

        df_first, df_second = self._splitter.split(dataframe)
        data_container.data[self.outputs[0]] = df_first
        data_container.data[self.outputs[1]] = df_second
        return data_container
