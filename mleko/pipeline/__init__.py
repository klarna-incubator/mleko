"""Pipeline subpackage for managing and executing data processing steps.

This subpackage provides the necessary components to create a customizable data processing pipeline. It includes
abstract base classes for pipeline steps, a flexible and customizable Pipeline class for managing and executing a
series of data processing steps, and a subpackage named `steps` that contains concrete implementations of the
`PipelineStep` class.
"""

from __future__ import annotations

from .data_container import DataContainer
from .pipeline import Pipeline
from .pipeline_step import PipelineStep


__all__ = ["Pipeline", "PipelineStep", "DataContainer"]
