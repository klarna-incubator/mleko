"""Module for fetching data from AWS S3 and storing it locally using the `S3Ingester` class."""
from __future__ import annotations

import json
import os
from concurrent import futures
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig as BotoTransferConfig
from botocore.config import Config as BotoConfig
from tqdm.auto import tqdm

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.file_helpers import clear_directory

from .base_ingester import BaseIngester


logger = CustomLogger()
"""A module-level custom logger."""


class S3Ingester(BaseIngester):
    """`S3Ingester` provides a convenient interface for fetching data from AWS S3 buckets and storing it locally.

    This class interacts with AWS S3 to download specified data from an S3 bucket.
    It supports manifest-based caching, enabling more efficient data fetching by verifying if the
    local dataset is up-to-date before downloading.
    """

    @auto_repr
    def __init__(
        self,
        destination_directory: str | Path,
        s3_bucket_name: str,
        s3_key_prefix: str,
        aws_profile_name: str | None = None,
        aws_region_name: str = "eu-west-1",
        num_workers: int = 64,
        manifest_file_name: str | None = "manifest",
        check_s3_timestamps: bool = True,
    ) -> None:
        """Initializes the S3 bucket client, configures the destination directory, and sets client-related parameters.

        Note:
            The S3 bucket client is initialized using the provided AWS profile and region. If no profile is provided,
            the default profile will be used. If no region is provided, the default region will be used.

            The profile and region is read from the AWS credentials file located at '~/.aws/credentials'.

        Args:
            destination_directory: Directory to store the fetched data locally.
            s3_bucket_name: Name of the S3 bucket containing the data.
            s3_key_prefix: Prefix of the S3 keys for the files to download.
            aws_profile_name: AWS profile name to use.
            aws_region_name: AWS region name where the S3 bucket is located.
            num_workers: Number of workers to use for concurrent downloads.
            manifest_file_name: Name of the manifest file.
            check_s3_timestamps: Whether to check if all S3 files have the same timestamp.

        Examples:
            >>> from mleko.dataset.sources import S3Ingester
            >>> s3_ingester = S3Ingester(
            ...     destination_directory="data",
            ...     s3_bucket_name="mleko-datasets",
            ...     s3_key_prefix="kaggle/ashishpatel26/indian-food-101",
            ...     aws_profile_name="mleko",
            ...     aws_region_name="eu-west-1",
            ...     num_workers=64,
            ...     manifest_file_name="manifest",
            ...     check_s3_timestamps=True,
            ... )
            >>> s3_ingester.fetch_data()
            [PosixPath('data/indian_food.csv')]
        """
        super().__init__(destination_directory)
        self._s3_bucket_name = s3_bucket_name
        self._s3_key_prefix = s3_key_prefix
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._s3_client = self._get_s3_client(self._aws_profile_name, self._aws_region_name)
        self._num_workers = num_workers
        self._manifest_file_name = manifest_file_name
        self._check_s3_timestamps = check_s3_timestamps

    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads the data from the S3 bucket and stores it in the 'destination_directory'.

        If 'force_recompute' is False, verifies whether the data in the local 'destination_directory' is current
        with the S3 bucket contents based on the manifest file, and skips downloading if it is up to date.

        Args:
            force_recompute: Whether to force the data source to recompute its output, even if it already exists.

        Raises:
            Exception: If files in the S3 bucket have different last modified dates, indicating potential corruption
                       or duplication.

        Returns:
            A list of Path objects pointing to the downloaded data files.
        """
        self._s3_client = self._get_s3_client(self._aws_profile_name, self._aws_region_name)
        resp = self._s3_client.list_objects(
            Bucket=self._s3_bucket_name,
            Prefix=self._s3_key_prefix,
        )

        if self._check_s3_timestamps:
            modification_dates = {key["LastModified"].day for key in resp["Contents"] if "LastModified" in key}
            if len(modification_dates) > 1:
                raise Exception(
                    "Files in S3 are from muliples dates. This might mean the data is corrupted/duplicated."
                )
        if self._manifest_file_name is not None:
            manifest_file_key = next(
                entry["Key"]
                for entry in resp["Contents"]
                if "Key" in entry and entry["Key"].endswith(self._manifest_file_name)
            )

            if not force_recompute and manifest_file_key:
                self._s3_client.download_file(
                    Bucket=self._s3_bucket_name,
                    Key=manifest_file_key,
                    Filename=str(self._destination_directory / self._manifest_file_name),
                )
                with open(self._destination_directory / self._manifest_file_name) as f:
                    manifest: dict[str, Any] = json.load(f)
                    if self._is_local_dataset_fresh(manifest):
                        logger.info(
                            "\033[32mCache Hit\033[0m: Local dataset is up to date with S3 bucket contents, "
                            "skipping download."
                        )
                        return self._get_local_filenames(["gz", "csv", "zip"])

        if force_recompute:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m: Downloading {self._s3_bucket_name}/{self._s3_key_prefix} to "
                f"{self._destination_directory} from S3."
            )
        else:
            logger.info(
                f"\033[31mCache Miss\033[0m: Downloading {self._s3_bucket_name}/{self._s3_key_prefix} to "
                f"{self._destination_directory} from S3."
            )

        clear_directory(self._destination_directory)
        keys_to_download = [
            entry["Key"]
            for entry in resp["Contents"]
            if "Key" in entry and (entry["Key"].endswith(".csv") or entry["Key"].endswith(".gz"))
        ]

        if keys_to_download:
            self._s3_fetch_all(keys_to_download)
            logger.info(f"Finished downloading {len(keys_to_download)} files from S3.")

        return self._get_local_filenames(["gz", "csv", "zip"])

    def _get_s3_client(  # type: ignore
        self,
        aws_profile_name: str | None,
        aws_region_name: str,
    ):
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
        client_config = BotoConfig(max_pool_connections=100)
        return boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=aws_region_name,
            config=client_config,
        )

    def _s3_fetch_file(self, key: str) -> None:
        """Downloads a single file from the S3 bucket to the destination specified in the constructor.

        Args:
            key: S3 key of the file to download.
        """
        gb = 1024**3
        transfer_config = BotoTransferConfig(
            use_threads=False,
            multipart_threshold=int(0.5 * gb),  # Multipart transfer if file > 500MB
        )
        file_path = self._destination_directory / Path(key).name
        with open(file_path, "wb") as data:
            self._s3_client.download_fileobj(
                Bucket=self._s3_bucket_name,
                Key=key,
                Fileobj=data,
                Config=transfer_config,
            )

    def _s3_fetch_all(self, keys: list[str]) -> None:
        """Downloads all specified files from the S3 bucket to the local directory concurrently.

        Args:
            keys: List of S3 keys for the files to download.
        """
        with tqdm(total=len(keys), desc="Downloading CSV files from S3") as pbar:
            with futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                for _ in executor.map(
                    self._s3_fetch_file,
                    keys,
                ):
                    pbar.update(1)

    def _is_local_dataset_fresh(self, manifest: dict[str, Any]) -> bool:
        """Checks if the local dataset is up-to-date with the manifest file.

        Args:
            manifest: Manifest file content as a dictionary.

        Returns:
            True if the local dataset is up-to-date, False otherwise.
        """
        for entry in manifest["entries"]:
            file_name: str = os.path.basename(entry["url"])
            file_size = int(entry["meta"]["content_length"])
            local_file_path = self._destination_directory / file_name
            if not local_file_path.exists() or file_size != os.path.getsize(local_file_path):
                return False
        return True
