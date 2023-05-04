"""This module contains the Pipeline class, which manages and executes a series of processing steps."""
from __future__ import annotations

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep


class Pipeline:
    """A class representing a pipeline of data processing steps.

    The pipeline is responsible for executing a series of `PipelineStep` instances in the order they are added.
    The output of each step is passed as input to the next step in the pipeline.
    """

    def __init__(self, steps: list[PipelineStep] | None = None) -> None:
        """Initialize the pipeline with a list of pipeline steps.

        Args:
            steps: An optional list of `PipelineStep` instances to be executed in the pipeline.
                   If not provided, an empty list will be used.
        """
        self.steps = steps if steps is not None else []

    def __repr__(self) -> str:
        """Returns a string representation of the Pipeline."""
        cls_name = type(self).__name__
        steps_str = "\n".join([f"  {index + 1}. {step!r}" for index, step in enumerate(self.steps)])
        return f"{cls_name}:\n{steps_str}"

    def add_step(self, step: PipelineStep) -> None:
        """Add a pipeline step to the end of the pipeline.

        Args:
            step: A `PipelineStep` instance to be added to the pipeline.
        """
        self.steps.append(step)

    def run(self, data_container: DataContainer | None = None) -> DataContainer:
        """Run the pipeline, executing each step in the order they were added.

        Args:
            data_container: Optional initial data to be passed as input to the first step in the pipeline.
                  If not provided, the first step's execute method should not require any input.

        Returns:
            The output of the last step in the pipeline.
        """
        if data_container is None:
            data_container = DataContainer()

        for step in self.steps:
            data_container = step.execute(data_container)

        return data_container
