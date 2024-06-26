"""Test suite for the `pipeline.pipeline` module."""

from __future__ import annotations

from pathlib import Path

from typing_extensions import TypedDict

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline import Pipeline
from mleko.pipeline.pipeline_step import PipelineStep


class TestPipeline:
    """Test suite for `pipeline.pipeline.Pipeline`."""

    class InputStep(PipelineStep):
        """Initial step."""

        class InputType(TypedDict):
            """The input type of the ConvertStep."""

            pass

        class OutputType(TypedDict):
            """The output type of the ConvertStep."""

            raw_data: str

        def execute(self, data_container: DataContainer, force_recompute: bool, disable_cache: bool) -> DataContainer:
            """Execute the step."""
            return DataContainer(data={"raw_data": [Path()]})

        def _get_input_model(self):
            return self.InputType

        def _get_output_model(self):
            return self.OutputType

    class AppendStep(PipelineStep):
        """Append data with another Path step."""

        class InputType(TypedDict):
            """The input type of the ConvertStep."""

            raw_data: str

        class OutputType(TypedDict):
            """The output type of the ConvertStep."""

            appended_data: str

        def execute(self, data_container: DataContainer, force_recompute: bool, disable_cache: bool) -> DataContainer:
            """Execute the step."""
            file_paths = data_container.data["raw_data"]
            if not isinstance(file_paths, list) or not all(isinstance(e, Path) for e in file_paths):
                raise ValueError
            data_container.data["appended_data"] = file_paths + [Path()]
            return data_container

        def _get_input_model(self):
            return self.InputType

        def _get_output_model(self):
            return self.OutputType

    def test_init(self):
        """Should successfully initialize the pipeline."""
        pipeline = Pipeline()
        assert not pipeline._steps

    def test_init_with_steps(self):
        """Should successfully initialize the pipeline with one or more PipelineStep instances."""
        step1 = self.InputStep(inputs={}, outputs={"raw_data": "raw_data"}, cache_group=None)
        step2 = self.AppendStep(
            inputs={"raw_data": "raw_data"}, outputs={"appended_data": "appended_data"}, cache_group=None
        )
        pipeline = Pipeline(steps=[step1, step2])

        assert len(pipeline._steps) == 2
        assert pipeline._steps[0] == step1
        assert pipeline._steps[1] == step2

    def test_repr(self):
        """Should represent Pipeline using representation of steps."""
        step1 = self.InputStep(inputs={}, outputs={"raw_data": "raw_data"}, cache_group=None)
        step2 = self.AppendStep(
            inputs={"raw_data": "raw_data"}, outputs={"appended_data": "appended_data"}, cache_group=None
        )
        pipeline = Pipeline(steps=[step1, step2])

        expected = f"Pipeline:\n  1. {step1!r}\n  2. {step2!r}"
        assert pipeline.__repr__() == expected

    def test_add_step(self):
        """Should successfully add new PipelineStep."""
        step1 = self.InputStep(inputs={}, outputs={"raw_data": "raw_data"}, cache_group=None)
        step2 = self.AppendStep(
            inputs={"raw_data": "raw_data"}, outputs={"appended_data": "appended_data"}, cache_group=None
        )
        pipeline = Pipeline(steps=[step1])
        pipeline.add_step(step2)

        assert len(pipeline._steps) == 2
        assert pipeline._steps[1] == step2

    def test_run(self):
        """Should run multiple PipelineSteps."""
        input_step = self.InputStep(inputs={}, outputs={"raw_data": "raw_data"}, cache_group=None)
        increment_step = self.AppendStep(
            inputs={"raw_data": "raw_data"}, outputs={"appended_data": "appended_data"}, cache_group=None
        )
        pipeline = Pipeline(steps=[input_step, increment_step])

        result = pipeline.run()

        assert isinstance(result, DataContainer)
        assert result.data["appended_data"] == [Path(), Path()]

    def test_run_empty_pipeline(self):
        """Should return empty DataContainer on bad PipelineStep."""
        pipeline = Pipeline()
        result = pipeline.run()

        assert isinstance(result, DataContainer)
        assert result.data == {}
