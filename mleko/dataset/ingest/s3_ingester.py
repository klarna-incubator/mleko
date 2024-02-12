"""Module for fetching data from AWS S3 and storing it locally using the `S3Ingester` class."""

from __future__ import annotations

import hashlib
import os
from concurrent import futures
from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatch
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig as BotoTransferConfig
from botocore.config import Config as BotoConfig
from tqdm.auto import tqdm

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_ingester import BaseIngester, LocalFileEntry, LocalManifestHandler


logger = CustomLogger()
"""A module-level custom logger."""


@dataclass
class S3FileManifest:
    """Manifest entry for a single S3 file."""

    key: Path
    """Key of the file, the full path of the file in the S3 bucket including the key prefix."""

    size: int
    """Size of the file in bytes."""

    last_modified: date
    """Last modified date of the file."""


class S3Ingester(BaseIngester):
    """`S3Ingester` provides a convenient interface for fetching data from AWS S3 buckets and storing it locally.

    This class interacts with AWS S3 to download specified data from an S3 bucket.
    It supports manifest-based caching, enabling more efficient data fetching by verifying if the
    local dataset is up-to-date before downloading.
    """

    @auto_repr
    def __init__(
        self,
        s3_bucket_name: str,
        s3_key_prefix: str,
        file_pattern: str | list[str] = "*",
        dataset_id: str | None = None,
        cache_directory: str | Path = "data/s3-ingester",
        aws_profile_name: str | None = None,
        aws_region_name: str = "eu-west-1",
        num_workers: int = 64,
        check_s3_timestamps: bool = True,
    ) -> None:
        """Initializes the S3 bucket client, configures the cache directory, and sets client-related parameters.

        Note:
            The S3 bucket client is initialized using the provided AWS profile and region. If no profile is provided,
            the default profile will be used. If no region is provided, the default region will be used.

            The profile and region is read from the AWS credentials file located at '~/.aws/credentials'.

        Args:
            s3_bucket_name: Name of the S3 bucket containing the data.
            s3_key_prefix: Prefix of the S3 keys for the files to download
            file_pattern: Pattern to match the files to download, e.g. `*.csv` or [`*.csv`, `*.json`], etc.
                For more information, see https://docs.python.org/3/library/fnmatch.html.
            dataset_id: Id of the dataset to be used instead of the default fingerprint (MD5 hash of the bucket
                name, key prefix, and region name). Note that this will overwrite any existing dataset with the same
                name in the cache directory, so make sure to use a unique name.
            cache_directory: Directory to store the fetched data locally.
            aws_profile_name: AWS profile name to use.
            aws_region_name: AWS region name where the S3 bucket is located.
            num_workers: Number of workers to use for concurrent downloads.
            check_s3_timestamps: Whether to check if all S3 files have the same timestamp.

        Examples:
            >>> from mleko.dataset.sources import S3Ingester
            >>> s3_ingester = S3Ingester(
            ...     s3_bucket_name="mleko-datasets",
            ...     s3_key_prefix="kaggle/ashishpatel26/indian-food-101",
            ...     file_pattern="file_*.csv",
            ...     dataset_id="indian_food", # Optional, but will store the data in "./data/indian_food/" instead of
            ...                               # "./data/<fingerprint>/".
            ...     aws_profile_name="mleko",
            ...     aws_region_name="eu-west-1",
            ...     num_workers=64,
            ...     check_s3_timestamps=True,
            ... )
            >>> s3_ingester.fetch_data()
            [PosixPath('data/indian_food/indian_food.csv')]
        """
        dataset_id = (
            dataset_id
            if dataset_id is not None
            else hashlib.md5((s3_bucket_name + s3_key_prefix + aws_region_name).encode()).hexdigest()
        )
        super().__init__(cache_directory, dataset_id)
        self._local_manifest_handler = LocalManifestHandler(
            self._cache_directory / f"{self._fingerprint}.manifest.json"
        )
        self._s3_bucket_name = s3_bucket_name
        self._s3_key_prefix = s3_key_prefix
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._s3_client = self._get_s3_client(self._aws_profile_name, self._aws_region_name)
        self._num_workers = num_workers
        self._check_s3_timestamps = check_s3_timestamps

        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]
        self._file_pattern = file_pattern

    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads the data from the S3 bucket and stores it in the 'cache_directory'.

        If 'force_recompute' is False, verifies whether the data in the local 'cache_directory' is current
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
        s3_manifest = self._build_s3_manifest()

        if self._check_s3_timestamps:
            modification_dates = {key.last_modified for key in s3_manifest}
            if len(modification_dates) > 1:
                error_msg = "Files in S3 are from muliples dates. This might mean the data is corrupted/duplicated."
                logger.error(error_msg)
                raise Exception(error_msg)

        if force_recompute:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m: Downloading files matching {self._file_pattern} from "
                f"{self._s3_bucket_name}/{self._s3_key_prefix} to {self._cache_directory} from S3."
            )
        else:
            if self._is_local_dataset_fresh(s3_manifest):
                logger.info(
                    "\033[32mCache Hit\033[0m: Local dataset is up to date with S3 bucket contents, "
                    "skipping download."
                )
                local_file_names = set(self._local_manifest_handler.get_file_names())
                s3_file_names: set[str] = {s3_file.key.name for s3_file in s3_manifest}
                files_to_delete = list(local_file_names.difference(s3_file_names))

                if len(files_to_delete) > 0:
                    logger.info(
                        f"Deleting {len(files_to_delete)} files from "
                        f"{self._cache_directory} that are no longer present in S3 or filtered out."
                    )

                self._delete_local_files(files_to_delete)
                self._local_manifest_handler.remove_files(files_to_delete)
                return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

            logger.info(
                f"\033[31mCache Miss\033[0m: Downloading {self._s3_bucket_name}/{self._s3_key_prefix} to "
                f"{self._cache_directory} from S3."
            )

        self._delete_local_files(self._local_manifest_handler.get_file_names())
        keys_to_download: list[str] = [str(s3_file.key) for s3_file in s3_manifest]
        if len(keys_to_download) > 0:
            self._s3_fetch_all(keys_to_download)
            self._local_manifest_handler.set_files(
                [
                    LocalFileEntry(name=Path(key).name, size=os.path.getsize(self._cache_directory / Path(key).name))
                    for key in keys_to_download
                ]
            )
            logger.info(f"Finished downloading {len(keys_to_download)} files from S3.")

        return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

    def _build_s3_manifest(self) -> list[S3FileManifest]:
        """Builds a manifest from the S3 response.

        The manifest contains the S3 keys, sizes, and last modified dates of the files in the S3 bucket.

        Raises:
            FileNotFoundError: If no files matching the file pattern are found in the S3 bucket.

        Returns:
            A list of `S3FileManifest` objects containing the S3 keys, sizes, and last modified dates of the files in
            the S3 bucket.
        """
        resp = self._s3_client.list_objects(
            Bucket=self._s3_bucket_name,
            Prefix=self._s3_key_prefix,
        )

        s3_manifest: list[S3FileManifest] = [
            S3FileManifest(key=Path(key["Key"]), size=key["Size"], last_modified=key["LastModified"].date())
            for key in resp.get("Contents", [])
            if "LastModified" in key
            and "Key" in key
            and "Size" in key
            and any(fnmatch(Path(key["Key"]).name, pattern) for pattern in self._file_pattern)
        ]

        if len(s3_manifest) == 0:
            msg = (
                f"No files matching {self._file_pattern} found in S3 bucket "
                f"{self._s3_bucket_name}/{self._s3_key_prefix}."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Found {len(s3_manifest)} file(s) matching any of {self._file_pattern} in S3 bucket.")

        return s3_manifest

    def _get_s3_client(
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
        file_path = self._cache_directory / Path(key).name
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
        with tqdm(total=len(keys), desc="Downloading files from S3") as pbar:
            with futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                for _ in executor.map(
                    self._s3_fetch_file,
                    keys,
                ):
                    pbar.update(1)

    def _is_local_dataset_fresh(self, s3_manifest: list[S3FileManifest]) -> bool:
        """Checks if the local dataset is up-to-date with the S3 manifest.

        Args:
            s3_manifest: Manifest built from S3 response.

        Returns:
            True if the local dataset is up-to-date, False otherwise.
        """
        for s3_file in s3_manifest:
            local_file_path = self._cache_directory / s3_file.key.name
            if not local_file_path.exists() or s3_file.size != os.path.getsize(local_file_path):
                return False
        return True
