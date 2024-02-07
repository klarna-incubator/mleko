"""Data management module for handling data used in the pipeline.

This module provides the DataContainer class, which serves as a common interface for various types of data,
enforcing shared structure and behavior. The goal is to facilitate data handling throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataContainer:
    """Class for holding data used in the pipeline.

    This class serves as a common interface and can be extended to enforce
    a shared structure or behavior across different types of data.

    Examples:
        >>> data_container = DataContainer()
        >>> data_container.data["my_data"] = [Path("path/to/data")]
        >>> data_container.data["my_data"]
        [Path("path/to/data")]
    """

    data: dict[str, Any] = field(default_factory=dict)
    """The data stored in the DataContainer.

    The data is stored in a dictionary, where the keys are the names of the data and the values are the data itself.
    """

    def __repr__(self) -> str:
        """Get string representation of DataContainer.

        Returns:
            String representation of the DataContainer, including data type and stored data.
        """
        data_type = type(self.data).__name__
        return f"<DataContainer: data_type={data_type}, data={self.data}>"
