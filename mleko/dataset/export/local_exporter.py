"""Module for exporting data to a local file using the `LocalExporter` class."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Literal, Union

from tqdm.auto import tqdm
from typing_extensions import TypedDict

from mleko.cache.fingerprinters.json_fingerprinter import JsonFingerprinter
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
from mleko.cache.handlers import write_joblib, write_json, write_pickle, write_string, write_vaex_dataframe
from mleko.utils import CustomLogger, LocalFileEntry, LocalManifestHandler, auto_repr

from .base_exporter import BaseExporter


logger = CustomLogger()
"""A module-level logger instance."""

ExportType = Literal["vaex", "json", "pickle", "joblib", "string"]
"""Type alias for the supported export types."""


class LocalExporterConfig(TypedDict):
    """Configuration for the LocalExporter."""

    destination: Union[str, Path]
    """The path of the file to which the data will be exported."""

    type: Literal["vaex", "json", "pickle", "joblib", "string"]
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
    def __init__(self, manifest_file_path: str | Path, delete_old_files: bool = False) -> None:
        """Initializes the `LocalExporter`.

        Note:
            The manifest is intended to be used to keep track of the exported file names and sizes. It should
            reflect the current state of the local dataset. In case a new set of files is exported and
            `delete_old_files` is set to True, the old files will be deleted unless they are present in the
            new data export.

        Args:
            manifest_file_path: Path to the manifest file to use for tracking exported files. If the file does
                not exist, it will be created.
            delete_old_files: Whether to delete the old files from the local dataset before exporting the new ones
                based on the manifest.
        """
        super().__init__()
        self._manifest_handler = LocalManifestHandler(manifest_file_path)
        self._delete_old_files = delete_old_files
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
            >>> exporter.export("test data", {"destination": "test.txt", "type": "string"})
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

        if not isinstance(config, list) and isinstance(data, list) and config["type"] != "json":
            msg = (
                "Data is a list, but the export type is not 'json'. Please provide a list of configs for a "
                "list of data items."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(config, list):
            data = [data]
            config = [config]

        previous_files = self._manifest_handler.get_file_names()
        results: list[Path] = []
        for d, c in tqdm(zip(data, config), desc="Exporting data", total=len(data)):
            results.append(self._export_single(d, c, force_recompute))

        if self._delete_old_files:
            untouched_files = list(set(previous_files) - set(map(str, results)))
            self._manifest_handler.remove_files(untouched_files)
            for file_name in map(Path, untouched_files):
                logger.warning(f"Removing old file: {file_name}")
                file_name.unlink()

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
        export_type = config["type"]
        destination = Path(config["destination"])
        self._ensure_path_exists(destination)

        current_file_entry = LocalFileEntry(
            name=str(destination),
            size=destination.stat().st_size if destination.exists() else 0,
            hash=self._hash_data(data, export_type),
        )
        manifest = self._manifest_handler._read_manifest()

        existing_entry = next((f for f in manifest.files if f.name == current_file_entry.name), None)

        if (
            not force_recompute
            and existing_entry is not None
            and existing_entry.hash == current_file_entry.hash
            and destination.exists()
        ):
            logger.info(f"\033[32mCache Hit\033[0m: Data already exported to {str(destination)!r}, skipping export.")
            return destination

        if force_recompute:
            logger.info(f"\033[33mForce Cache Refresh\033[0m: Forcing data export to {str(destination)!r}.")
        else:
            logger.info(f"\033[31mCache Miss\033[0m: Exporting data to {str(destination)!r}.")

        self._run_export_function(data, destination, export_type)
        current_file_entry.size = destination.stat().st_size
        self._manifest_handler.remove_files([current_file_entry.name])
        self._manifest_handler.add_files([current_file_entry])

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
