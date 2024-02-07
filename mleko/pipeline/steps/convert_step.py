"""Module for data conversion as a part of the pipeline process.

This module contains the `ConvertStep` class which is a specialized pipeline step for handling data format
conversion. It uses the provided `BaseDataConverter` for converting the data into the desired format.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from typing_extensions import TypedDict, TypeVar

from mleko.dataset.convert import BaseConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr


logger = CustomLogger()
"""The logger for the module."""

TypedDictType = TypeVar("T", bound=TypedDict)  # type: ignore
"""Type variable for TypedDict type annotations."""


class ConvertStepInputType(TypedDict):
    """The input type of the ConvertStep."""

    file_paths: Union[str, List[Path], List[str]]
    """List of file paths or the key identifying the list of file paths to be converted."""


class ConvertStepOutputType(TypedDict):
    """The output type of the ConvertStep."""

    data_schema: str
    """The key for the DataSchema after conversion."""

    dataframe: str
    """The key for the DataFrame after conversion."""


class ConvertStep(PipelineStep):
    """Pipeline step that manages data conversion from one format to another."""

    _inputs: ConvertStepInputType
    _outputs: ConvertStepOutputType

    @auto_repr
    def __init__(
        self,
        converter: BaseConverter,
        inputs: ConvertStepInputType,
        outputs: ConvertStepOutputType,
        cache_group: str | None = None,
    ) -> None:
        """Initialize the ConvertStep with the specified data converter.

        Args:
            converter: The DataConverter responsible for handling data format conversion.
            inputs: A dictionary of input keys following the `ConvertStepInputType` schema.
            outputs: A dictionary of output keys following the `ConvertStepOutputType` schema.
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
            ValueError: If the input data types are invalid.

        Returns:
            A DataContainer containing the result.
        """
        file_paths = self._validate_and_get_input(self._inputs["file_paths"], list, data_container)
        if not all(isinstance(e, Path) or isinstance(e, str) for e in file_paths):
            msg = "Invalid data type: {type(file_paths)}. Expected List[Path] or List[str]."
            logger.error(msg)
            raise ValueError(msg)

        data_schema, dataframe = self._converter.convert(file_paths, self._cache_group, force_recompute)
        data_container.data[self._outputs["data_schema"]] = data_schema
        data_container.data[self._outputs["dataframe"]] = dataframe
        return data_container

    def _get_input_model(self) -> type[ConvertStepInputType]:
        """Get the input type for the ConvertStep.

        Returns:
            Input type for the ConvertStep.
        """
        return ConvertStepInputType

    def _get_output_model(self) -> type[ConvertStepOutputType]:
        """Get the output type for the ConvertStep.

        Returns:
            Output type for the ConvertStep.
        """
        return ConvertStepOutputType
