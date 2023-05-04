"""This module defines the abstract base class for pipeline steps.

A PipelineStep is a unit of work in a data processing pipeline.
Each step should have a specific purpose and should be able to run independently or as part of a larger pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from mleko.pipeline.data_container import DataContainer


class PipelineStep(ABC):
    """A base class for all pipeline steps, providing a standardized interface for implementing data processing.

    The `PipelineStep` class is an abstract base class (ABC),
    which ensures that any subclass of this class must provide an implementation for the `execute` method.
    This method is the core of each pipeline step, responsible for executing the data processing operation.
    """

    @abstractmethod
    def execute(self, data_container: DataContainer) -> DataContainer:
        """Execute the data processing operation associated with this pipeline step.

        Args:
            data_container: The input data to be processed by this step.
                The format and structure of this data will depend on the specific pipeline step implementation.

        Raises:
            NotImplementedError: Must be implemented by subclass.

        Returns:
            The processed data after the operation is performed. The format and structure of this data will
                depend on the specific pipeline step implementation. The `DataContainer` contains the data and
                an enum determining the data type.
        """
        raise NotImplementedError
