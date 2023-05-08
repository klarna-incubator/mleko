"""This module defines the abstract base class for pipeline steps in a data processing pipeline.

The module provides a standard interface for implementing data processing steps as part of a larger pipeline,
via the `PipelineStep` abstract base class. Each `PipelineStep` subclass should have a specific purpose and
should be able to run independently or as part of the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from mleko.pipeline.data_container import DataContainer


class PipelineStep(ABC):
    """Base class for all pipeline steps, ensuring the standardized interface for performing data processing operations.

    Descendants of this class must implement the `execute` method, which carries out the data processing operation
    related to the step.
    """

    @abstractmethod
    def execute(self, data_container: DataContainer) -> DataContainer:
        """Execute the data processing operation associated with this pipeline step.

        The input data to be processed is passed via `data_container`, and the structure depends on the specific
        implementation. After the operation, the method returns the processed data in the form of a `DataContainer`,
        which includes the data and an enum determining the data type.

        Args:
            data_container: Input data for this step's processing operation.

        Raises:
            NotImplementedError: Must be implemented by subclass.

        Returns:
            Processed data as a `DataContainer`.
        """
        raise NotImplementedError
