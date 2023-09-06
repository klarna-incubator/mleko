"""Module handling dataframe splitting.

This module contains the SplitStep class, which is responsible for splitting a DataFrame into two parts.
It uses a BaseDataSplitter to perform the actual splitting.
"""
from __future__ import annotations

from vaex import DataFrame

from mleko.dataset.split.base_splitter import BaseSplitter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class SplitStep(PipelineStep):
    """Pipeline step that splits a DataFrame into two parts."""

    _num_inputs = 1
    """Number of inputs expected by the SplitStep."""

    _num_outputs = 2
    """Number of outputs expected by the SplitStep."""

    @auto_repr
    def __init__(
        self,
        splitter: BaseSplitter,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the SplitStep with the specified data splitter.

        Args:
            splitter: The DataSplitter responsible for handling data splitting.
            inputs: List or tuple of input keys expected by this step. Should contain a single key,
                corresponding to the DataFrame to be split.
            outputs: List or tuple of output keys produced by this step. Should contain two keys,
                corresponding to the two DataFrames after splitting.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._splitter = splitter

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform data splitting using the configured splitter.

        Args:
            data_container: Contains the DataFrame to be split.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the split data as two vaex DataFrames.
        """
        dataframe = data_container.data[self._inputs[0]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"Invalid data type: {type(dataframe)}. Expected vaex DataFrame.")

        df1, df2 = self._splitter.split(dataframe, self._cache_group, force_recompute)
        data_container.data[self._outputs[0]] = df1
        data_container.data[self._outputs[1]] = df2
        return data_container
