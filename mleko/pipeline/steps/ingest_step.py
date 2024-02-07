"""Module handling data ingestion from a data source in the pipeline process.

This module contains the `IngestStep` class which is a specialized pipeline step designed for handling data
fetching from a specified `BaseIngester`. It's responsible for retrieving data from the data source, and
returning a `DataContainer` object containing the list of fetched files.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from mleko.dataset.ingest import BaseIngester
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils import auto_repr


class IngestStepInputType(TypedDict):
    """The input type of the IngestStep."""

    pass


class IngestStepOutputType(TypedDict):
    """The output type of the IngestStep."""

    file_paths: str
    """The key for the list of fetched file paths."""


class IngestStep(PipelineStep):
    """Pipeline step that manages data ingestion from a configured data source."""

    _inputs: IngestStepInputType
    _outputs: IngestStepOutputType

    @auto_repr
    def __init__(
        self,
        ingester: BaseIngester,
        inputs: IngestStepInputType,
        outputs: IngestStepOutputType,
    ) -> None:
        """Initialize the IngestStep with the specified data source.

        Args:
            ingester: The data source from which to fetch the data, a BaseIngester instance.
            inputs: A dictionary of input keys following the `IngestStepInputType` schema.
            outputs: A dictionary of output keys following the `IngestStepOutputType` schema.
        """
        super().__init__(inputs, outputs, None)
        self._ingester = ingester

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Fetch data from the configured data source and return a DataContainer with fetched files.

        Args:
            data_container: Input data for this step's processing operation.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            A DataContainer containing the result.
        """
        files = self._ingester.fetch_data(force_recompute)
        data_container.data[self._outputs["file_paths"]] = files
        return data_container

    def _get_input_model(self) -> type[IngestStepInputType]:
        """Get the input type for the IngestStep.

        Returns:
            Input type for the IngestStep.
        """
        return IngestStepInputType

    def _get_output_model(self) -> type[IngestStepOutputType]:
        """Get the output type for the IngestStep.

        Returns:
            Output type for the IngestStep.
        """
        return IngestStepOutputType
