"""Module for exporting data to AWS S3 from the local filesystem using the `S3Exporter` class."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Optional

from tqdm.auto import tqdm
from typing_extensions import TypedDict

from mleko.utils import CustomLogger, S3Client, auto_repr
from mleko.utils.s3_helpers import S3FileManifest

from .base_exporter import BaseExporter


logger = CustomLogger()
"""A module-level custom logger."""


class S3ExporterConfig(TypedDict):
    """Configuration for the S3 exporter."""

    bucket_name: str
    """Name of the S3 bucket to export the files to."""

    key_prefix: str
    """Key prefix (folder) to place the files under."""

    extra_args: Optional[dict[str, Any]]
    """Extra arguments to pass to the S3 client.

    Refer to the `boto3` documentation for the `upload_file` method for more information.
    """


class S3Exporter(BaseExporter):
    """`S3Exporter` provides functionality to export files to an S3 bucket from the local filesystem.

    The class interacts with AWS S3 using the `boto3` library to upload files to an S3 bucket. It supports
    multi-threaded uploads to improve performance and caching to avoid re-uploading files that already exist in the
    destination.
    """

    @auto_repr
    def __init__(
        self,
        manifest_file_name: str | None = "manifest",
        max_concurrent_files: int = 64,
        workers_per_file: int = 1,
        aws_profile_name: str | None = None,
        aws_region_name: str = "eu-west-1",
    ) -> None:
        """Initializes the `S3Exporter` class and creates the S3 client.

        Note:
            The S3 bucket client is initialized using the provided AWS profile and region. If no profile is provided,
            the default profile will be used. If no region is provided, the default region will be used.

            The profile and region is read from the AWS credentials file located at '~/.aws/credentials'.

        Note:
            If you want to update the S3 bucket content extra arguments, make sure to set the `force_recompute`
            parameter to `True` when calling the `export` method. This will force the exporter to re-upload the
            files to the S3 bucket with the updated extra arguments.

        Warning:
            The `max_concurrent_files` and `workers_per_file` parameters are used to control the
            number of concurrent upload and parts upload per file, respectively. These parameters should be
            set based on the available system resources and the S3 bucket's performance limits. The total number of
            concurrent threads is the product of these two parameters
            (i.e., `max_concurrent_files * workers_per_file`).

        Args:
            manifest_file_name: Name of the manifest file to store the S3 file metadata.
            max_concurrent_files: Maximum number of files to upload concurrently.
            workers_per_file: Number of parts to upload concurrently for each file. This is useful for
                upload large files faster, as it allows for parallel upload of different parts of the file.
            aws_profile_name: AWS profile name to use.
            aws_region_name: AWS region name where the S3 bucket is located.

        Examples:
            >>> from mleko.dataset.export import S3Exporter
            >>> s3_exporter = S3Exporter()
            >>> s3_exporter.export(["file1.csv", "file2.csv"], {"bucket_name": "my-bucket", "key_prefix": "data/"})
            ['s3://my-bucket/data/file1.csv', 's3://my-bucket/data/file2.csv']
        """
        super().__init__()
        self._manifest_file_name = manifest_file_name
        self._max_concurrent_files = max_concurrent_files
        self._workers_per_file = workers_per_file
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._s3_client = S3Client(self._aws_profile_name, self._aws_region_name)

    def export(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, data: list[Path] | list[str], config: S3ExporterConfig, force_recompute: bool = False
    ) -> list[str]:
        """Export the files to the specified S3 bucket and key prefix.

        Will verify if the files already exist in the S3 bucket and key prefix before exporting them.

        Args:
            data: List of file paths to export to S3.
            config: Configuration for the S3 exporter.
            force_recompute: If set to True, forces the data to be exported even if it already exists
                at the destination.

        Returns:
            List of S3 paths to the exported files.
        """
        file_paths = [Path(file_path) for file_path in data]
        bucket_name = config["bucket_name"]
        key_prefix = config["key_prefix"]
        extra_args = config["extra_args"]

        self._s3_client.refresh_client()
        s3_manifest = self._s3_client.get_s3_manifest(
            bucket_name=bucket_name,
            key_prefix=key_prefix,
            manifest_file_name=self._manifest_file_name,
            file_pattern="*",
        )

        s3_path_string = f"s3://{Path(bucket_name) / key_prefix}/"
        if force_recompute:
            logger.info(f"\033[33mForce Cache Refresh\033[0m: Uploading {len(file_paths)} files to {s3_path_string}.")
        else:
            if self._s3_client.is_s3_dataset_up_to_date(file_paths, s3_manifest):
                logger.info(f"\033[32mCache Hit\033[0m: Files already uploaded to {s3_path_string}, skipping export.")

                if self._manifest_file_name is not None:
                    file_names = [file_path.name for file_path in file_paths]
                    s3_manifest = [
                        s3_file_manifest for s3_file_manifest in s3_manifest if s3_file_manifest.key.name in file_names
                    ]
                    self._s3_client.put_s3_manifest(
                        bucket_name,
                        f"{Path(key_prefix) / self._manifest_file_name}",
                        s3_manifest,
                    )

                return [f"s3://{bucket_name / s3_file_manifest.key}" for s3_file_manifest in s3_manifest]

            logger.info(f"\033[31mCache Miss\033[0m: Uploading {len(file_paths)} files to {s3_path_string}.")

        s3_manifest = self._s3_export_all(file_paths, bucket_name, key_prefix, extra_args)
        if self._manifest_file_name is not None:
            self._s3_client.put_s3_manifest(
                bucket_name,
                f"{Path(key_prefix) / self._manifest_file_name}",
                s3_manifest,
            )
        logger.info(f"Finished uploading {len(file_paths)} files to {s3_path_string}.")
        return [f"s3://{bucket_name / s3_file_manifest.key}" for s3_file_manifest in s3_manifest]

    def _s3_export_all(
        self, file_paths: list[Path], bucket_name: str, key_prefix: str, extra_args: dict[str, Any] | None
    ) -> list[S3FileManifest]:
        """Exports all files to S3 to the specified bucket and key prefix in parallel.

        Args:
            file_paths: List of file paths to export to S3.
            bucket_name: Name of the S3 bucket to export the files to.
            key_prefix: Key prefix to use for the S3 object keys.
            extra_args: Extra arguments to pass to the S3 client.

        Returns:
            S3 manifest for the exported files.
        """
        s3_manifest = []
        with tqdm(total=len(file_paths), desc="Exporting to S3") as pbar:
            with ThreadPoolExecutor(max_workers=min(len(file_paths), self._max_concurrent_files)) as executor:
                for s3_file_manifest in executor.map(
                    self._s3_client.upload_file,
                    file_paths,
                    repeat(bucket_name),
                    repeat(key_prefix),
                    repeat(extra_args),
                    repeat(self._workers_per_file),
                ):
                    s3_manifest.append(s3_file_manifest)
                    pbar.update(1)

        return s3_manifest
