"""Docstring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import vaex


@dataclass
class DataContainer:
    """Class for data used in the pipeline.

    This class serves as a common interface and can be used to enforce
    a shared structure or behavior across different types of data.

    Attributes:
        data: The stored data.
    """

    data: list[Path] | vaex.DataFrame | None = None

    def __repr__(self) -> str:
        """Get string representation of DataContainer.

        Returns:
            String representation of DataContainer.
        """
        data_type = type(self.data).__name__
        return f"<DataContainer: data_type={data_type}, data={self.data}>"
