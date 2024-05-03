"""The subpackage provides functionality for exporting data to various destinations.

This subpackage contains classes designed to easily export to different destinations, like AWS S3 or locally,

The following ingester classes are provided by the subpackage:
    - `BaseExporter`: An abstract base class for data export classes.
    - `LocalExporter`: A class for exporting data to a local file. Supports exporting data in various formats,
        such as CSV, Arrow, JSON, and Pickle.
"""

from __future__ import annotations

from .base_exporter import BaseExporter
from .local_exporter import LocalExporter


__all__ = ["BaseExporter", "LocalExporter"]
