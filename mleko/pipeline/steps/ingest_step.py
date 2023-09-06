"""Module handling data ingestion from a data source in the pipeline process.

This module contains the `IngestStep` class which is a specialized pipeline step designed for handling data
fetching from a specified `BaseIngester`. It's responsible for retrieving data from the data source, and
returning a `DataContainer` object containing the list of fetched files.
"""
from __future__ import annotations

from mleko.dataset.ingest import BaseIngester
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
        ingester: BaseIngester,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize the IngestStep with the specified data source.

        Args:
            ingester: The data source from which to fetch the data, a BaseIngester instance.
            inputs: List or tuple of input keys expected by this step. Should be empty.
            outputs: List or tuple of output keys produced by this step. Should contain a single key,
                corresponding to the list of fetched file Paths.
        """
        super().__init__(inputs, outputs, None)
        self._ingester = ingester

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Fetch data from the configured data source and return a DataContainer with fetched files.

        Args:
            data_container: Input data for this step's processing operation.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Returns:
            DataContainer: A DataContainer containing a list of fetched files.
        """
        files = self._ingester.fetch_data(force_recompute)
        data_container.data[self._outputs[0]] = files
        return data_container
