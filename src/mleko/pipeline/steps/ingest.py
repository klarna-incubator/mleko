"""Module handling data ingestion from a data source in the pipeline process.

This module contains the `IngestStep` class which is a specialized pipeline step designed for handling data
fetching from a specified `BaseDataSource`. It's responsible for retrieving data from the data source, and
returning a `DataContainer` object containing the list of fetched files.
"""
from __future__ import annotations

from mleko.data.sources import BaseDataSource
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils import auto_repr


class IngestStep(PipelineStep):
    """Pipeline step that manages data ingestion from a configured data source."""

    _num_inputs = 0
    """Number of inputs expected by the IngestStep."""

    _num_outputs = 1
    """Number of outputs expected by the IngestStep."""

    @auto_repr
    def __init__(
        self,
        data_source: BaseDataSource,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize the IngestStep with the specified data source.

        Args:
            data_source: The data source from which to fetch the data, a BaseDataSource instance.
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
        """
        super().__init__(inputs, outputs)
        self._data_source = data_source

    def execute(self, _data_container: DataContainer) -> DataContainer:
        """Fetch data from the configured data source and return a DataContainer with fetched files.

        The `_data_container` parameter is unused in this step as this operation only deals with data ingestion
        and no input data is required.

        Returns:
            DataContainer: A DataContainer containing a list of fetched files.
        """
        files = self._data_source.fetch_data()
        _data_container.data[self.outputs[0]] = files
        return _data_container
