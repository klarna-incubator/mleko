"""Abstract base class module for data exporter implementations to store data to various destinations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from typing_extensions import TypedDict


TypedDictType = TypeVar("T", bound=TypedDict)  # type: ignore
"""Type variable for TypedDict type annotations."""


class BaseExporter(ABC):
    """`BaseExporter` is an abstract base class for data exporters."""

    @abstractmethod
    def export(
        self,
        data: Any,
        exporter_config: dict[str, Any] | TypedDictType,  # type: ignore
        force_recompute: bool = False,
    ) -> str | Path:
        """Exports the data to a destination.

        Args:
            data: Data to be exported.
            exporter_config: Configuration for the export destination.
            force_recompute: If set to True, forces the data to be exported even if it already exists
                at the destination.
        """
        raise NotImplementedError
