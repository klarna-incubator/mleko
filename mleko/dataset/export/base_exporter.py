"""Abstract base class module for data exporter implementations to store data to various destinations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseExporter(ABC):
    """`BaseExporter` is an abstract base class for data exporters."""

    @abstractmethod
    def export(
        self,
        data: Any | list[Any],
        config: dict[str, Any] | list[dict[str, Any]],
        force_recompute: bool = False,
    ) -> str | Path | list[str] | list[Path] | None:
        """Exports the data to a destination.

        Args:
            data: Data to be exported.
            config: Configuration for the export destination.
            force_recompute: If set to True, forces the data to be exported even if it already exists
                at the destination.
        """
        raise NotImplementedError
