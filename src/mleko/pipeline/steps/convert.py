"""Module for data conversion as a part of the pipeline process.

This module contains the `ConvertStep` class which is a specialized pipeline step for handling data format
conversion. It uses the provided `BaseDataConverter` for converting the data into the desired format.
"""
from __future__ import annotations

from pathlib import Path

from mleko.data.converters import BaseDataConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class ConvertStep(PipelineStep):
    """Pipeline step that manages data conversion from one format to another."""

    @auto_repr
    def __init__(self, converter: BaseDataConverter) -> None:
        """Initialize the ConvertStep with the specified data converter.

        Args:
            converter: The DataConverter responsible for handling data format conversion.
        """
        super().__init__()
        self._converter = converter

    def execute(self, data_container: DataContainer) -> DataContainer:
        """Perform data format conversion using the configured converter.

        Args:
            data_container: Contains a list of file Paths to be converted.

        Raises:
            ValueError: If data container contains invalid data - not a list of Paths.

        Returns:
            A DataContainer containing the converted data as a vaex dataframe.
        """
        if not isinstance(data_container.data, list) or not all(isinstance(e, Path) for e in data_container.data):
            raise ValueError

        df = self._converter.convert(data_container.data)
        return DataContainer(data=df)
