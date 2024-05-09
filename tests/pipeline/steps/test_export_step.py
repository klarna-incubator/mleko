"""Test suite for the `pipeline.steps.export_step` module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.export import BaseExporter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.pipeline.steps.export_step import ExportStep


class TestExportStep:
    """Test suite for `pipeline.steps.export_step.ExportStep`."""

    def test_init(self):
        """Should initialize ExportStep instance."""
        exporter_mock = MagicMock(spec=BaseExporter)
        export_step = ExportStep(
            exporter=exporter_mock,
            inputs={"data": DataSchema(), "export_config": {"file_name": "data_schema.json"}},
            outputs={"file_path": "raw_data"},
        )

        assert isinstance(export_step, PipelineStep)
        assert export_step._exporter == exporter_mock

    def test_execute(self):
        """Should execute the ExportStep and return the fetched data as DataContainer."""
        exporter_mock = MagicMock(spec=BaseExporter)
        exporter_mock.export.return_value = Path("file_1.txt")

        export_step = ExportStep(
            exporter=exporter_mock,
            inputs={"data": DataSchema(), "export_config": {"file_name": "data_schema.json"}},
            outputs={"file_path": "raw_data"},
        )
        result = export_step.execute(DataContainer(), force_recompute=False, disable_cache=False)

        exporter_mock.export.assert_called_once()
        assert isinstance(result, DataContainer)
        assert result.data["raw_data"] == Path("file_1.txt")

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        exporter = MagicMock(spec=BaseExporter)
        with pytest.raises(ValueError):
            ExportStep(exporter=exporter, inputs={"input": "test"}, outputs={"file_paths": "raw_data"})  # type: ignore

        with pytest.raises(ValueError):
            ExportStep(exporter=exporter, inputs={"input": "test"}, outputs={})  # type: ignore
