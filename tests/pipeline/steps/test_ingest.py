"""Test suite for the `pipeline.steps.ingest` module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mleko.data.sources import BaseDataSource
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.pipeline.steps.ingest import IngestStep


class TestIngestStep:
    """Test suite for `pipeline.steps.ingest.IngestStep`."""

    def test_init(self):
        """Should initialize IngestStep instance."""
        ds_mock = MagicMock(spec=BaseDataSource)
        ingest_step = IngestStep(data_source=ds_mock, outputs=["raw_data"])

        assert isinstance(ingest_step, PipelineStep)
        assert ingest_step._data_source == ds_mock

    def test_execute(self):
        """Should execute the IngestStep and return the fetched data as DataContainer."""
        ds_mock = MagicMock(spec=BaseDataSource)
        ds_mock.fetch_data.return_value = [Path("file_1.txt"), Path("file_2.txt")]

        ingest_step = IngestStep(data_source=ds_mock, outputs=["raw_data"])
        result = ingest_step.execute(DataContainer())

        ds_mock.fetch_data.assert_called_once()
        assert isinstance(result, DataContainer)
        assert result.data["raw_data"] == [Path("file_1.txt"), Path("file_2.txt")]

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        data_source = MagicMock(spec=BaseDataSource)
        with pytest.raises(ValueError):
            IngestStep(data_source=data_source, inputs=[], outputs=["converted_data", "raw_data"])

        with pytest.raises(ValueError):
            IngestStep(data_source=data_source, inputs=["raw_data"], outputs=[])
