"""Module defining the IngestStep class for data ingestion."""
from __future__ import annotations

from mleko.data.sources import BaseDataSource
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class IngestStep(PipelineStep):
    """IngestStep is a pipeline step responsible for ingesting data from a specified DataSource."""

    @auto_repr
    def __init__(self, data_source: BaseDataSource) -> None:
        """Initialize the IngestStep with the specified BaseDataSource.

        Args:
            data_source: The BaseDataSource from which to ingest data.
        """
        self._data_source = data_source

    def execute(self, _data_container: DataContainer) -> DataContainer:
        """Ingest the data from the specified BaseDataSource.

        The `_data_container` parameter is unused in this step.

        Returns:
            A list of Path objects to fetched files.
        """
        files = self._data_source.fetch_data()
        return DataContainer(data=files)
