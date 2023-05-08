"""Data management module for handling data used in the pipeline.

This module provides the DataContainer class, which serves as a common interface for various types of data,
enforcing shared structure and behavior. The goal is to facilitate data handling throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import vaex


@dataclass
class DataContainer:
    """Class for holding data used in the pipeline.

    This class serves as a common interface and can be extended to enforce
    a shared structure or behavior across different types of data.

    Attributes:
        data: The stored data.
    """

    data: list[Path] | vaex.DataFrame | None = None

    def __repr__(self) -> str:
        """Get string representation of DataContainer.

        Returns:
            String representation of the DataContainer, including data type and stored data.
        """
        data_type = type(self.data).__name__
        return f"<DataContainer: data_type={data_type}, data={self.data}>"
