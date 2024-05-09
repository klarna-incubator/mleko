"""Module handling data exporting to a destination in the pipeline process.

This module contains the `ExportStep` class which is a specialized pipeline step designed for handling data
exporting from a specified `BaseExporter`. It's responsible for exporting the data to a destination, and
returning a `DataContainer` object containing the destination path of the exported data.
"""

from __future__ import annotations

from typing import Any, Union

from typing_extensions import TypedDict

from mleko.dataset.export import BaseExporter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils import auto_repr


class ExportStepInputType(TypedDict):
    """The input type of the ExportStep."""

    data: Union[str, Any]
    """The data to export."""

    export_config: Union[str, dict]
    """The configuration for the export operation."""


class ExportStepOutputType(TypedDict):
    """The output type of the ExportStep."""

    file_path: str
    """The path of the exported file."""


class ExportStep(PipelineStep):
    """Pipeline step that manages data exporting to a destination using a specified `BaseExporter`."""

    _inputs: ExportStepInputType
    _outputs: ExportStepOutputType

    @auto_repr
    def __init__(
        self,
        exporter: BaseExporter,
        inputs: ExportStepInputType,
        outputs: ExportStepOutputType,
    ) -> None:
        """Initialize the IngestStep with the specified `BaseExporter` and input/output keys.

        Args:
            exporter: The `BaseExporter` instance to use for exporting data.
            inputs: A dictionary of input keys following the `IngestStepInputType` schema.
            outputs: A dictionary of output keys following the `IngestStepOutputType` schema.
        """
        super().__init__(inputs, outputs, None)
        self._exporter = exporter

    def execute(self, data_container: DataContainer, force_recompute: bool, disable_cache: bool) -> DataContainer:
        """Export the data to the destination using the specified `BaseExporter`.

        Args:
            data_container: Input data for this step's processing operation.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.
            disable_cache: Not used for ingestion steps.

        Returns:
            A DataContainer containing the result.
        """
        data = (
            data_container.data[self._inputs["data"]] if isinstance(self._inputs["data"], str) else self._inputs["data"]
        )
        exporter_config = self._validate_and_get_input(self._inputs["export_config"], dict, data_container)

        files = self._exporter.export(data, exporter_config, force_recompute)
        data_container.data[self._outputs["file_path"]] = files
        return data_container

    def _get_input_model(self) -> type[ExportStepInputType]:
        """Get the input type for the ExportStep.

        Returns:
            Input type for the ExportStep.
        """
        return ExportStepInputType

    def _get_output_model(self) -> type[ExportStepOutputType]:
        """Get the output type for the ExportStep.

        Returns:
            Output type for the ExportStep.
        """
        return ExportStepOutputType
