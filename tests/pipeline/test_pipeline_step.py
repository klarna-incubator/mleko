"""Test suite for the `pipeline.pipeline_step` module."""
from __future__ import annotations

from pathlib import Path

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep


class TestPipelineStep:
    """Test suite for `pipeline.pipeline_step.PipelineStep`."""

    class DummyPipelineStep(PipelineStep):
        """A dummy implementation of PipelineStep for testing purposes."""

        _num_inputs = 0
        _num_outputs = 3

        def execute(self, data_container: DataContainer):
            """Execute the dummy step."""
            data_container.data[self.outputs[0]] = "first"  # type: ignore
            data_container.data[self.outputs[2]] = "second"  # type: ignore
            data_container.data[self.outputs[2]] = "third"  # type: ignore
            return data_container

    def test_init(self):
        """Should init a concrete PipelineStep subclass."""
        concrete_step = self.DummyPipelineStep(inputs=[], outputs=["first", "second", "third"])
        assert isinstance(concrete_step, PipelineStep)

    def test_execute(self):
        """Should successfully implementation and execution of the `execute` method in a PipelineStep subclass."""
        dummy_data = [Path()]
        input_data_container = DataContainer(data={"raw_data": dummy_data})
        concrete_step = self.DummyPipelineStep(inputs=[], outputs=["first", "second", "third"])
        output_data_container = concrete_step.execute(input_data_container)
        assert output_data_container.data == input_data_container.data
