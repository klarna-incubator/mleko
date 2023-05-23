"""Test suite for the `pipeline.pipeline_step` module."""
from __future__ import annotations

from pathlib import Path

import pytest
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep


class TestPipelineStep:
    """Test suite for `pipeline.pipeline_step.PipelineStep`."""

    class DummyPipelineStep(PipelineStep):
        """A dummy implementation of PipelineStep for testing purposes."""

        _num_inputs = 1
        _num_outputs = 3

        def execute(self, data_container: DataContainer):
            """Execute the dummy step."""
            file_paths = data_container.data[self.inputs[0]]
            if not isinstance(file_paths, list) or not all(isinstance(e, Path) for e in file_paths):
                raise ValueError

            data_container.data[self.outputs[0]] = "0"  # type: ignore
            data_container.data[self.outputs[1]] = "1"  # type: ignore
            data_container.data[self.outputs[2]] = "2"  # type: ignore
            return data_container

    def test_init(self):
        """Should init a concrete PipelineStep subclass."""
        concrete_step = self.DummyPipelineStep(inputs=["input"], outputs=["first", "second", "third"])
        assert isinstance(concrete_step, PipelineStep)

    def test_execute(self):
        """Should successfully implementation and execution of the `execute` method in a PipelineStep subclass."""
        dummy_data = [Path()]
        input_data_container = DataContainer(data={"input": dummy_data})
        concrete_step = self.DummyPipelineStep(inputs=["input"], outputs=["first", "second", "third"])
        output_data_container = concrete_step.execute(input_data_container)
        assert output_data_container.data == input_data_container.data

    def test_execute_with_invalid_input(self):
        """Should raise a ValueError if the input data is invalid."""
        input_data_container = DataContainer(data={"input": "invalid"})  # type: ignore
        concrete_step = self.DummyPipelineStep(inputs=["input"], outputs=["first", "second", "third"])
        with pytest.raises(ValueError):
            concrete_step.execute(input_data_container)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        with pytest.raises(ValueError):
            self.DummyPipelineStep(inputs=["input1", "input2"], outputs=["first", "second", "third"])

        with pytest.raises(ValueError):
            self.DummyPipelineStep(inputs=["input"], outputs=["first", "second"])
