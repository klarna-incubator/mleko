"""Module containing the FilterStep class."""

from __future__ import annotations

from typing import Union

from typing_extensions import TypedDict
from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.filter.base_filter import BaseFilter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class FilterStepInputType(TypedDict):
    """The input type of the FilterStep."""

    data_schema: Union[str, DataSchema]
    """The key for the DataSchema."""

    dataframe: Union[str, DataFrame]
    """The key for the DataFrame to be filtered."""


class FilterStepOutputType(TypedDict):
    """The output type of the FilterStep."""

    dataframe: str
    """The key for the filtered DataFrame."""


class FilterStep(PipelineStep):
    """Pipeline step that filters a DataFrame."""

    _inputs: FilterStepInputType
    _outputs: FilterStepOutputType

    @auto_repr
    def __init__(
        self,
        filter: BaseFilter,
        inputs: FilterStepInputType,
        outputs: FilterStepOutputType,
        cache_group: str | None = None,
    ) -> None:
        """Initialize the FilterStep with the specified data filter.

        Args:
            filter: The DataFilter responsible for handling data filtering.
            inputs: A dictionary of input keys following the `FilterStepInputType` schema.
            outputs: A dictionary of output keys following the `FilterStepOutputType` schema.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._filter = filter

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform data filtering using the configured filter.

        Args:
            data_container: Contains the DataFrame to be filtered.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the result.
        """
        data_schema = self._validate_and_get_input(self._inputs["data_schema"], DataSchema, data_container)
        dataframe = self._validate_and_get_input(self._inputs["dataframe"], DataFrame, data_container)

        filtered_dataframe = self._filter.filter(data_schema, dataframe, self._cache_group, force_recompute)
        data_container.data[self._outputs["dataframe"]] = filtered_dataframe
        return data_container

    def _get_input_model(self) -> type[FilterStepInputType]:
        """Get the input type for the FilterStep.

        Returns:
            Input type for the FilterStep.
        """
        return FilterStepInputType

    def _get_output_model(self) -> type[FilterStepOutputType]:
        """Get the output type for the FilterStep.

        Returns:
            Output type for the FilterStep.
        """
        return FilterStepOutputType
