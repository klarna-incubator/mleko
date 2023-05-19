"""Provides a flexible and customizable Pipeline class for managing and executing a series of data processing steps.

The module defines a Pipeline class that allows for the creation of a sequence of data processing steps. It is
designed for building complex data processing workflows by chaining together custom `PipelineStep` instances.
The Pipeline class encapsulates the ordered sequence of steps added to it and provides methods to manage, execute,
and visualize these steps. Each step's output is passed as input to the next step, effectively managing the flow
of data through the processing pipeline.
"""
from __future__ import annotations

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""A CustomLogger instance that's used throughout the module for logging."""


class Pipeline:
    """Encapsulates a pipeline that manages and executes a series of data processing steps in a defined order."""

    def __init__(self, steps: list[PipelineStep] | None = None) -> None:
        """Creates a new Pipeline instance, initializing it with a list of steps or an empty list.

        Args:
            steps: An optional list of `PipelineStep` instances that define the data processing steps in the
                   pipeline. If not provided, the pipeline will be initialized with an empty list of steps,
                   allowing steps to be added later using the `add_step` method.
        """
        self.steps = steps if steps is not None else []

    def __repr__(self) -> str:
        """Returns a string representation of the Pipeline, including the ordered list of steps.

        Returns:
            String representaition of Pipeline that includes the class name and each step in the order they appear
            in the pipeline, numbered for easier identification.
        """
        cls_name = type(self).__name__
        steps_str = "\n".join([f"  {index + 1}. {step!r}" for index, step in enumerate(self.steps)])
        return f"{cls_name}:\n{steps_str}"

    def add_step(self, step: PipelineStep) -> None:
        """Appends a new PipelineStep to the end of the pipeline, extending the processing sequence.

        Adding a step to the pipeline implies that it will be executed after all the steps previously
        appended to the pipeline when calling the `run` method.

        Args:
            step: The PipelineStep instance to be added at the end of the pipeline's steps list.
        """
        self.steps.append(step)

    def run(self, data_container: DataContainer | None = None) -> DataContainer:
        """Executes the pipeline steps in the order they were added, passing output from one to the next.

        Processes the initial given data or an empty data container through each step in the pipeline.
        The output of each step is passed as input to the next step, allowing the given input to be transformed
        through the whole sequence of steps.

        Args:
            data_container: An optional DataContainer instance carrying the input data to be processed by the
                            first step in the pipeline. If not provided, an empty DataContainer instance will be
                            created automatically, and the first step's execute method must handle it.

        Returns:
            The output as a DataContainer instance from the last step in the pipeline after processing the data.
        """
        if data_container is None:
            data_container = DataContainer()

        for i, step in enumerate(self.steps):
            logger.info(f"Executing step {i+1}: {step.__class__.__name__}")
            data_container = step.execute(data_container)
            logger.info(f"Finished step {i+1}")
        return data_container
