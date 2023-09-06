"""Module for data conversion as a part of the pipeline process.

This module contains the `ConvertStep` class which is a specialized pipeline step for handling data format
conversion. It uses the provided `BaseDataConverter` for converting the data into the desired format.
"""
from __future__ import annotations

from pathlib import Path

from mleko.dataset.convert import BaseConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class ConvertStep(PipelineStep):
    """Pipeline step that manages data conversion from one format to another."""

    _num_inputs = 1
    """Number of inputs expected by the ConvertStep."""

    _num_outputs = 2
    """Number of outputs expected by the ConvertStep."""

    @auto_repr
    def __init__(
        self,
        converter: BaseConverter,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the ConvertStep with the specified data converter.

        Args:
            converter: The DataConverter responsible for handling data format conversion.
            inputs: List or tuple of input keys expected by this step. Should contain a single key,
                corresponding to the list of file Paths to be converted.
            outputs: List or tuple of output keys produced by this step. Should contain two keys,
                corresponding to the DataSchema and DataFrame after conversion.
            cache_group: The cache group to use.
        """
        super().__init__(inputs, outputs, cache_group)
        self._converter = converter

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform data format conversion using the configured converter.

        Args:
            data_container: Contains a list of file Paths to be converted.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a list of Paths.

        Returns:
            A DataContainer containing the DataSchema and DataFrame after conversion.
        """
        file_paths = data_container.data[self._inputs[0]]
        if not isinstance(file_paths, list) or not all(isinstance(e, Path) for e in file_paths):
            raise ValueError(f"Invalid data type: {type(file_paths)}. Expected list of Paths.")

        data_schema, dataframe = self._converter.convert(file_paths, self._cache_group, force_recompute)
        data_container.data[self._outputs[0]] = data_schema
        data_container.data[self._outputs[1]] = dataframe
        return data_container
