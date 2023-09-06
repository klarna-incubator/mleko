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

    Note:
        The _num_inputs and _num_outputs attributes are used to validate the number of inputs and outputs, respectively,
        for each step. These attributes are set by the subclasses, and should not be modified by the user.
        When implementing a new step, you should set these attributes to the expected number of inputs and outputs.
    """

    _num_inputs: int
    """Number of inputs expected by the PipelineStep."""

    _num_outputs: int
    """Number of outputs expected by the PipelineStep."""

    def __init__(
        self,
        inputs: list[str] | tuple[str, ...] | tuple[()],
        outputs: list[str] | tuple[str, ...] | tuple[()],
        cache_group: str | None,
    ) -> None:
        """Initialize a new PipelineStep with the provided input and output keys.

        Args:
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
            cache_group: The cache group to use.
        """
        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)
        self._cache_group = cache_group

        self._validate_inputs()
        self._validate_outputs()

    @abstractmethod
    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Execute the data processing operation associated with this pipeline step.

        The `execute` method is the main entry point for the data processing operation associated with this step.
        It receives a `DataContainer` instance as input, containing the data to be processed by this step.
        The method should perform the processing operation and return the processed data as a `DataContainer` instance.

        Args:
            data_container: Input data for this step's processing operation.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def _validate_inputs(self) -> None:
        """Check the input keys for compliance with this step's requirements.

        Raises:
            ValueError: If the PipelineStep has an invalid number of inputs.
        """
        if len(self._inputs) != self._num_inputs:
            raise ValueError(f"{self.__class__.__name__} must have exactly {self._num_inputs} input(s).")

    def _validate_outputs(self) -> None:
        """Check the output keys for compliance with this step's requirements.

        Raises:
            ValueError: If the PipelineStep has an invalid number of outputs.
        """
        if len(self._outputs) != self._num_outputs:
            raise ValueError(f"{self.__class__.__name__} must have exactly {self._num_outputs} output(s).")
