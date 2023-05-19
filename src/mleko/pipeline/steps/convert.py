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

    _num_inputs = 1
    """Number of inputs expected by the ConvertStep."""

    _num_outputs = 1
    """Number of outputs expected by the ConvertStep."""

    @auto_repr
    def __init__(
        self,
        converter: BaseDataConverter,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize the ConvertStep with the specified data converter.

        Args:
            converter: The DataConverter responsible for handling data format conversion.
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
        """
        super().__init__(inputs, outputs)
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
        file_paths = data_container.data[self.inputs[0]]
        if not isinstance(file_paths, list) or not all(isinstance(e, Path) for e in file_paths):
            raise ValueError

        df = self._converter.convert(file_paths)
        data_container.data[self.outputs[0]] = df
        return data_container
