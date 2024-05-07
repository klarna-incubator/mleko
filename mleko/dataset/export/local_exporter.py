"""Module for exporting data to a local file using the `LocalExporter` class."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Literal, Union

from typing_extensions import TypedDict

from mleko.cache.fingerprinters.json_fingerprinter import JsonFingerprinter
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.cache.handlers import write_joblib, write_json, write_pickle, write_string, write_vaex_dataframe
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_exporter import BaseExporter


logger = CustomLogger()
"""A module-level logger instance."""

ExportType = Literal["vaex", "json", "pickle", "joblib", "string"]
"""Type alias for the supported export types."""


class LocalExporterConfig(TypedDict):
    """Configuration for the LocalExporter."""

    export_destination: Union[str, Path]
    """The path of the file to which the data will be exported."""

    export_type: Literal["vaex", "json", "pickle", "joblib", "string"]
    """The type of export to perform. Supported types are 'vaex', 'json', 'pickle', 'joblib', and 'string'.

    Note:
        - 'vaex' is used for exporting Vaex DataFrames, which are exported using the Arrow format or
            the CSV format depending on the file extension.
        - 'json' is used for exporting JSON data.
        - 'pickle' is used for exporting generic data using Pickle.
        - 'joblib' is used for exporting generic data using Joblib.
        - 'string' is used for exporting string data.
    """


class LocalExporter(BaseExporter):
    """`LocalExporter` class for exporting data to a local file.

    This class provides methods for exporting data to a local file using various methods, such as CSV, Arrow, JSON,
    and Pickle. It can be chained with other exporters to export data to multiple destinations, such
    as Python -> Local -> S3.
    """

    @auto_repr
    def __init__(self) -> None:
        """Initializes the LocalExporter."""
        super().__init__()
        self._exporters: dict[ExportType, Callable[[Path, Any], None]] = {
            "vaex": write_vaex_dataframe,
            "json": write_json,
            "pickle": write_pickle,
            "joblib": write_joblib,
            "string": write_string,
        }

    def export(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: Any | list[Any],
        config: LocalExporterConfig | list[LocalExporterConfig],
        force_recompute: bool = False,
    ) -> list[Path]:
        """Exports the data to a local file.

        Args:
            data: Data to be exported.
            config: Configuration for the export destination following the `LocalExporterConfig` schema.
            force_recompute: If set to True, forces the data to be exported even if it already exists on disk.

        Examples:
            >>> from mleko.dataset.export import LocalExporter
            >>> exporter = LocalExporter()
            >>> exporter.export("test data", {"export_destination": "test.txt", "export_type": "string"})
            Path('test.txt')
        """
        if isinstance(config, list) and not isinstance(data, list):
            msg = "Data is not a list, but the config is a list. Please provide a single config for a single data item."
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(config, list) and isinstance(data, list) and len(config) != len(data):
            msg = "Number of data items and number of configs do not match."
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(config, list) and isinstance(data, list) and config["export_type"] != "json":
            msg = (
                "Data is a list, but the export type is not 'json'. Please provide a list of configs for a "
                "list of data items."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(config, list):
            data = [data]
            config = [config]

        results: list[Path] = []
        for d, c in zip(data, config):
            results.append(self._export_single(d, c, force_recompute))

        return results

    def _export_single(self, data: Any, config: LocalExporterConfig, force_recompute: bool) -> Path:
        """Exports a single data item to a local file.

        Args:
            data: Data to be exported.
            config: Configuration for the export destination following the `LocalExporterConfig` schema.
            force_recompute: If set to True, forces the data to be exported even if it already exists on disk.

        Returns:
            The path to the exported file.
        """
        export_type = config["export_type"]
        destination = config["export_destination"]
        if isinstance(destination, str):
            destination = Path(destination)

        self._ensure_path_exists(destination)
        suffix = destination.suffix

        hash_destination = destination.with_suffix(suffix + ".hash")
        data_hash = self._hash_data(data, export_type)
        if (
            not force_recompute
            and hash_destination.exists()
            and hash_destination.read_text() == data_hash
            and destination.exists()
        ):
            logger.info(f"\033[32mCache Hit\033[0m: Data already exported to {str(destination)!r}, skipping export.")
            return destination

        if force_recompute:
            logger.info(f"\033[33mForce Cache Refresh\033[0m: Forcing data export to {str(destination)!r}.")
        else:
            logger.info(f"\033[31mCache Miss\033[0m: Exporting data to {str(destination)!r}.")

        self._run_export_function(data, destination, export_type)
        hash_destination.write_text(data_hash)

        return destination

    def _ensure_path_exists(self, path: Path) -> None:
        """Ensures the specified path exists.

        Args:
            path: The path to ensure exists.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    def _run_export_function(self, data: Any, destination: Path, export_type: ExportType) -> None:
        exporter = self._exporters.get(export_type)
        if exporter is None:
            msg = f"Unsupported data type: {type(data)} for the export type: {export_type}."
            logger.error(msg)
            raise ValueError(msg)

        exporter(destination, data)

    def _hash_data(self, data: Any, export_type: ExportType) -> str:
        """Generates a hash for the given data.

        Args:
            data: Data to generate a hash for.

        Returns:
            A hash of the data.
        """
        if export_type == "vaex":
            return VaexFingerprinter().fingerprint(data)
        if export_type == "json":
            return JsonFingerprinter().fingerprint(data)
        if export_type == "string":
            return hashlib.md5(str(data).encode()).hexdigest()
        return hashlib.md5(pickle.dumps(data)).hexdigest()
