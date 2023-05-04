"""Test suite for the `pipeline.pipeline` module."""
from __future__ import annotations

from pathlib import Path

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline import Pipeline
from mleko.pipeline.pipeline_step import PipelineStep


class TestPipeline:
    """Test suite for `pipeline.pipeline.Pipeline`."""

    class InputStep(PipelineStep):
        """Initial step."""

        def execute(self, _data_container: DataContainer) -> DataContainer:
            """Execute the step."""
            return DataContainer(data=[Path()])

    class AppendStep(PipelineStep):
        """Append data with another Path step."""

        def execute(self, data_container: DataContainer) -> DataContainer:
            """Execute the step."""
            if not isinstance(data_container.data, list) or not all(isinstance(e, Path) for e in data_container.data):
                raise ValueError
            new_data = data_container.data + [Path()]
            return DataContainer(data=new_data)

    def test_init(self):
        """Should successfully initialize the pipeline."""
        pipeline = Pipeline()
        assert not pipeline.steps

    def test_init_with_steps(self):
        """Should successfully initialize the pipeline with one or more PipelineStep instances."""
        step1 = self.InputStep()
        step2 = self.AppendStep()
        pipeline = Pipeline(steps=[step1, step2])

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0] == step1
        assert pipeline.steps[1] == step2

    def test_repr(self):
        """Should represent Pipeline using representation of steps."""
        step1 = self.InputStep()
        step2 = self.AppendStep()
        pipeline = Pipeline(steps=[step1, step2])

        expected = f"Pipeline:\n  1. {step1!r}\n  2. {step2!r}"
        assert pipeline.__repr__() == expected

    def test_add_step(self):
        """Should successfully add new PipelineStep."""
        step1 = self.InputStep()
        step2 = self.AppendStep()
        pipeline = Pipeline(steps=[step1])
        pipeline.add_step(step2)

        assert len(pipeline.steps) == 2
        assert pipeline.steps[1] == step2

    def test_run(self):
        """Should run multiple PipelineSteps."""
        input_step = self.InputStep()
        increment_step = self.AppendStep()
        pipeline = Pipeline(steps=[input_step, increment_step])

        result = pipeline.run()

        assert isinstance(result, DataContainer)
        assert result.data == [Path(), Path()]

    def test_run_empty_pipeline(self):
        """Should return empty DataContainer on bad PipelineStep."""
        pipeline = Pipeline()
        result = pipeline.run()

        assert isinstance(result, DataContainer)
        assert result.data is None
