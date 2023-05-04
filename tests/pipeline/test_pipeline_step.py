"""Test suite for the `pipeline.pipeline_step` module."""
from __future__ import annotations

from pathlib import Path

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep


class TestPipelineStep:
    """Test suite for `pipeline.pipeline_step.PipelineStep`."""

    class DummyPipelineStep(PipelineStep):
        """A dummy implementation of PipelineStep for testing purposes."""

        def execute(self, data_container: DataContainer):
            """Execute the dummy step."""
            return DataContainer(data_container.data)

    def test_init(self):
        """Should init a concrete PipelineStep subclass."""
        concrete_step = self.DummyPipelineStep()
        assert isinstance(concrete_step, PipelineStep)

    def test_execute(self):
        """Should successfully implementation and execution of the `execute` method in a PipelineStep subclass."""
        dummy_data = [Path()]
        input_data_container = DataContainer(dummy_data)
        concrete_step = self.DummyPipelineStep()
        output_data_container = concrete_step.execute(input_data_container)
        assert output_data_container.data == input_data_container.data
