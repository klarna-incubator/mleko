"""Module defining the ConvertStep class for data conversion."""
from __future__ import annotations

from pathlib import Path

from mleko.data.converters import BaseDataConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class ConvertStep(PipelineStep):
    """ConvertStep is a pipeline step responsible for converting data from one format to another."""

    @auto_repr
    def __init__(self, converter: BaseDataConverter) -> None:
        """Initialize the ConvertStep with the specified DataConverter.

        Args:
            converter: The DataConverter that will convert data formats.
        """
        super().__init__()
        self._converter = converter

    def execute(self, data_container: DataContainer) -> DataContainer:
        """Convert the data format.

        Args:
            data_container: File Paths to read from.

        Raises:
            ValueError: Invalid data container, should contain a list of Path instances.

        Returns:
            A vaex dataframe on a converted format.
        """
        if not isinstance(data_container.data, list) or not all(isinstance(e, Path) for e in data_container.data):
            raise ValueError

        df = self._converter.convert(data_container.data)
        return DataContainer(data=df)
