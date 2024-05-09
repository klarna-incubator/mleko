"""This module contains helper functions for working with AWS S3."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig as BotoTransferConfig
from botocore.config import Config as BotoConfig

from .custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level custom logger."""


@dataclass
class S3FileManifest:
    """Manifest entry for a single S3 file."""

    key: Path
    """Key of the file, the full path of the file in the S3 bucket including the key prefix."""

    size: int
    """Size of the file in bytes."""

    last_modified: datetime.datetime
    """Last modified date of the file."""


class S3Client:
    """Helper class for working with AWS S3."""

    def __init__(
        self,
        aws_profile_name: str | None = None,
        aws_region_name: str = "eu-west-1",
    ) -> None:
        """Initializes an S3 client with the specified AWS profile and region.

        Args:
            aws_profile_name: AWS profile name to use. Defaults to None.
            aws_region_name: AWS region name where the S3 bucket is located.
        """
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._client = S3Client.get_s3_client(self._aws_profile_name, self._aws_region_name)

    @staticmethod
    def get_s3_client(aws_profile_name: str | None, aws_region_name: str):
        """Creates an S3 client using the provided AWS profile and region.

        Args:
            aws_profile_name: AWS profile name to use.
            aws_region_name: AWS region name where the S3 bucket is located.

        Returns:
            An S3 client configured with the specified profile and region.
        """
        credentials = boto3.Session(
            region_name=aws_region_name,
            profile_name=aws_profile_name,
        ).get_credentials()

        if credentials is None:
            msg = "AWS credentials not found. Please ensure that your AWS credentials are correctly set up."
            logger.error(msg)
            raise ValueError(msg)

        client_config = BotoConfig(max_pool_connections=100)
        return boto3.client(
            "s3",  # type: ignore
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=aws_region_name,
            config=client_config,
        )

    def refresh_client(self) -> None:
        """Refreshes the S3 client in case the credentials have changed."""
        self._client = S3Client.get_s3_client(self._aws_profile_name, self._aws_region_name)

    def download_file(
        self,
        destination_directory: Path,
        bucket_name: str,
        key: str,
        num_workers: int = 1,
        multipart_threshold_gb: float = 0.5,
    ) -> Path:
        """Downloads a file from S3 and saves it to the destination directory.

        Args:
            destination_directory: Destination directory where the file should be saved.
            bucket_name: Name of the S3 bucket.
            key: Key of the file to fetch.
            num_workers: Number of workers to use for downloading the file. Set to 1 for single-threaded download.
            multipart_threshold_gb: Threshold in GB for multipart transfer. If the file size is greater than this
                threshold, the file will be downloaded using multipart transfer.

        Returns:
            Path where the file is saved.
        """
        file_path = destination_directory / Path(key).name
        self._client.download_file(
            Bucket=bucket_name,
            Key=key,
            Filename=str(file_path),
            Config=self._get_boto_transfer_config(num_workers, multipart_threshold_gb),
        )
        return file_path

    def upload_file(
        self,
        file_path: Path,
        bucket_name: str,
        key_prefix: str,
        extra_args: dict[str, Any] | None = None,
        num_workers: int = 1,
        multipart_threshold_gb: float = 0.5,
    ) -> S3FileManifest:
        """Uploads a file to S3.

        Args:
            file_path: Path to the file to upload.
            bucket_name: Name of the S3 bucket.
            key_prefix: Key prefix to use for the file in the S3 bucket. The file will be uploaded to
                `f"{key_prefix}/{file_path.name}"`.
            extra_args: Extra arguments to pass to the S3 client.
            num_workers: Number of workers to use for uploading the file. Set to 1 for single-threaded upload.
            multipart_threshold_gb: Threshold in GB for multipart transfer. If the file size is greater than this
                threshold, the file will be uploaded using multipart transfer.

        Returns:
            S3 manifest entry for the uploaded file.
        """
        file_key = Path(key_prefix) / file_path.name
        self._client.upload_file(
            Filename=str(file_path),
            Bucket=bucket_name,
            Key=str(file_key),
            ExtraArgs=extra_args,
            Config=self._get_boto_transfer_config(num_workers, multipart_threshold_gb),
        )
        return S3FileManifest(key=file_key, size=file_path.stat().st_size, last_modified=datetime.datetime.now())

    def read_object(self, bucket_name: str, key: str) -> bytes:
        """Reads the contents of a object from S3.

        Args:
            bucket_name: The name of the S3 bucket.
            key: Key of the object to read.

        Returns:
            Contents of the object as bytes.
        """
        obj = self._client.get_object(Bucket=bucket_name, Key=key)
        return obj["Body"].read()

    def write_object(self, bucket_name: str, key: str, body: bytes | str) -> str:
        """Writes an object to S3.

        Args:
            bucket_name: The name of the S3 bucket.
            key: Key of the object to write.
            body: Body of the object to write.
        """
        self._client.put_object(Bucket=bucket_name, Key=key, Body=body)
        return f"s3://{Path(bucket_name) / key}"

    def get_s3_manifest(
        self,
        bucket_name: str,
        key_prefix: str,
        manifest_file_name: str | None = "manifest",
        file_pattern: str | list[str] = "*",
    ) -> list[S3FileManifest]:
        """Gets the S3 manifest for the files in the S3 bucket.

        Args:
            bucket_name: Name of the S3 bucket.
            key_prefix: Key prefix to the files in the S3 bucket.
            manifest_file_name: Optional name of a manifest file located on S3. If provided, the manifest from S3 will
                be used to determine the files to include, before applying the file pattern.
            file_pattern: Pattern to match the files to download, e.g. `*.csv` or [`*.csv`, `*.json`], etc.
                For more information, see https://docs.python.org/3/library/fnmatch.html.

        Raises:
            FileNotFoundError: If no files matching the file pattern are found in the S3 bucket.

        Returns:
            A list of `S3FileManifest` objects containing the S3 keys, sizes, and last modified dates of the files in
            the S3 bucket.
        """
        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]

        s3_contents = [
            entry
            for entry in self._client.list_objects(
                Bucket=bucket_name,
                Prefix=key_prefix,
            ).get("Contents", [])
            if "LastModified" in entry and "Key" in entry and "Size" in entry
        ]

        if manifest_file_name is not None:
            manifest_file_key = next(
                (entry["Key"] for entry in s3_contents if entry["Key"].endswith(manifest_file_name)), None
            )

            if manifest_file_key:
                manifest: set[str] = {
                    entry["url"].split(key_prefix)[-1].lstrip("/")
                    for entry in json.loads(S3Client().read_object(bucket_name, manifest_file_key)).get("entries", [])
                    if "url" in entry
                }

                s3_contents = [
                    entry for entry in s3_contents if entry["Key"].split(key_prefix)[-1].lstrip("/") in manifest
                ]

        s3_manifest: list[S3FileManifest] = [
            S3FileManifest(key=Path(entry["Key"]), size=entry["Size"], last_modified=entry["LastModified"])
            for entry in s3_contents
            if any(fnmatch(entry["Key"].split(key_prefix)[-1].lstrip("/"), pattern) for pattern in file_pattern)
        ]

        return s3_manifest

    def put_s3_manifest(self, bucket_name: str, key: str, s3_manifest: list[S3FileManifest]) -> None:
        """Puts a S3 manifest to the specified key in the S3 bucket.

        Args:
            bucket_name: Name of the S3 bucket.
            key: Key of the manifest file in the S3 bucket.
            s3_manifest: S3 manifest to write to the S3 bucket.
        """
        s3_manifest_dict = {
            "entries": [
                {
                    "url": str(s3_file_manifest.key),
                    "meta": {"content_length": s3_file_manifest.size},
                }
                for s3_file_manifest in s3_manifest
            ]
        }
        self.write_object(bucket_name, key, json.dumps(s3_manifest_dict, indent=2))

    def is_local_dataset_up_to_date(self, local_directory: Path | str, s3_manifest: list[S3FileManifest]) -> bool:
        """Checks if the local dataset is up to date with the S3 manifest.

        Args:
            local_directory: Local directory where the files are stored.
            s3_manifest: S3 manifest containing the files to compare.

        Returns:
            True if the local dataset is up to date with the S3 manifest, False otherwise.
        """
        local_directory = Path(local_directory)
        for s3_file in s3_manifest:
            local_file = local_directory / s3_file.key.name
            if not local_file.exists() or local_file.stat().st_size != s3_file.size:
                return False
        return True

    def is_s3_dataset_up_to_date(
        self,
        local_directory: list[Path] | list[str],
        s3_manifest: list[S3FileManifest],
    ) -> bool:
        """Checks if the S3 dataset is up to date with the local files.

        Args:
            local_directory: Local directory where the files are stored.
            s3_manifest: S3 manifest containing the files to compare.

        Returns:
            True if the S3 dataset is up to date with the local files, False otherwise.
        """
        local_files = [Path(file_path) for file_path in local_directory]
        for file_path in local_files:
            s3_file_manifest = next((entry for entry in s3_manifest if entry.key.name == file_path.name), None)
            if s3_file_manifest is None or file_path.stat().st_size != s3_file_manifest.size:
                return False
        return True

    def _get_boto_transfer_config(
        self,
        num_workers: int,
        multipart_threshold_gb: float,
    ) -> BotoTransferConfig:
        """Returns a Boto transfer configuration based on the number of workers and multipart threshold.

        Args:
            num_workers: Number of workers to use for downloading the file.
            multipart_threshold_gb: Threshold in GB for multipart transfer.

        Returns:
            Boto transfer configuration.
        """
        return BotoTransferConfig(
            use_threads=True if num_workers > 1 else False,
            max_concurrency=num_workers,
            multipart_threshold=int(multipart_threshold_gb * 1024**3),
        )
