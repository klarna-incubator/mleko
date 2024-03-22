"""Module containing the SplitStep class."""

from __future__ import annotations

from typing import Union

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.split.base_splitter import BaseSplitter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class SplitStepInputType(TypedDict):
    """The input type of the SplitStep."""

    dataframe: Union[str, DataFrame]
    """DataFrame or the key for the DataFrame to be split."""


class SplitStepOutputType(TypedDict):
    """The output type of the SplitStep."""

    dataframe_1: str
    """The key for the first DataFrame after splitting."""

    dataframe_2: str
    """The key for the second DataFrame after splitting."""


class SplitStep(PipelineStep):
    """Pipeline step that splits a DataFrame into two parts."""

    _inputs: SplitStepInputType
    _outputs: SplitStepOutputType

    @auto_repr
    def __init__(
        self,
        splitter: BaseSplitter,
        inputs: SplitStepInputType,
        outputs: SplitStepOutputType,
        cache_group: str | None = None,
    ) -> None:
        """Initialize the SplitStep with the specified data splitter.

        Args:
            splitter: The DataSplitter responsible for handling data splitting.
            inputs: A dictionary of input keys following the `SplitStepInputType` schema.
            outputs: A dictionary of output keys following the `SplitStepOutputType` schema.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._splitter = splitter

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform data splitting using the configured splitter.

        Args:
            data_container: Contains the DataFrame to be split.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the result.
        """
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        df1, df2 = self._splitter.split(dataframe, self._cache_group, force_recompute)
        data_container.data[self._outputs["dataframe_1"]] = df1
        data_container.data[self._outputs["dataframe_2"]] = df2
        return data_container

    def _get_input_model(self) -> type[SplitStepInputType]:
        """Get the input type for the SplitStep.

        Returns:
            Input type for the SplitStep.
        """
        return SplitStepInputType

    def _get_output_model(self) -> type[SplitStepOutputType]:
        """Get the output type for the SplitStep.

        Returns:
            Output type for the SplitStep.
        """
        return SplitStepOutputType
