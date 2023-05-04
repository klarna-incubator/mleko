"""Docstring."""
from __future__ import annotations

import glob
import json
import os
from abc import ABC, abstractmethod
from concurrent import futures
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig as BotoTransferConfig
from botocore.config import Config as BotoConfig
from tqdm import tqdm

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.file_helpers import clear_directory


logger = CustomLogger()


class BaseDataSource(ABC):
    """Abstract base class for data sources used to fetch data."""

    def __init__(self, destination_dir: str | Path) -> None:
        """Initializes the data source and creates the destination directory if it does not exist.

        Args:
            destination_dir: Destination directory to store the fetched data.
        """
        self._destination_dir = Path(destination_dir)
        self._destination_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_data(self, use_cache: bool = True) -> list[Path]:
        """Fetch the data to the `destination_dir` using the configured data source.

        Args:
            use_cache: If available for the child class, will skip the data fetching if up-to-date data exists
                inside the `destination_dir`. Defaults to True.

        Raises:
            NotImplementedError: Needs to be implemented by the class inheriting the `BaseDataSource`.

        Returns:
            List of Path objects to the data.
        """
        raise NotImplementedError


class S3DataSource(BaseDataSource):
    """A class to handle fetching data from an AWS S3 bucket and storing it locally."""

    @auto_repr
    def __init__(
        self,
        destination_dir: str | Path,
        s3_bucket_name: str,
        s3_key_prefix: str,
        aws_profile_name: str = "default",
        aws_region_name: str = "eu-west-1",
        num_workers: int = 64,
        manifest_file_name: str = "manifest",
        check_s3_timestamps: bool = True,
    ) -> None:
        """Initializes the S3 bucket client and prepares the destination directory.

        Args:
            destination_dir: Destination directory to store the fetched data.
            s3_bucket_name: Name of the S3 bucket containing the data.
            s3_key_prefix: Prefix of the S3 keys for the files to download.
            aws_profile_name: AWS profile name to use. Defaults to "default".
            aws_region_name: AWS region name where the S3 bucket is located. Defaults to "eu-west-1".
            num_workers: Number of workers to use for concurrent downloads. Defaults to 64.
            manifest_file_name: Name of the manifest file. Defaults to "manifest".
            check_s3_timestamps: Whether to check if all S3 files have the same timestamp. Defaults to True.
        """
        super().__init__(destination_dir=destination_dir)

        self._s3_bucket_name = s3_bucket_name
        self._s3_key_prefix = s3_key_prefix
        self._s3_client = self._get_s3_client(aws_profile_name, aws_region_name)

        self._num_workers = num_workers

        self._manifest_file_name = manifest_file_name
        self._check_s3_timestamps = check_s3_timestamps

    def fetch_data(self, use_cache: bool = True) -> list[Path]:
        """Fetches the data from the S3 bucket to the `destination_dir`.

        Args:
            use_cache: If True, checks if the local dataset is up-to-date with the S3 bucket contents
                using the manifest file, and skips downloading if it is. Defaults to True.

        Raises:
            Exception: If the files in the S3 bucket are from multiple dates,
                which might indicate data corruption or duplication.

        Returns:
            List of Path objects of downloaded data.
        """
        resp = self._s3_client.list_objects(
            Bucket=self._s3_bucket_name,
            Prefix=self._s3_key_prefix,
        )

        if self._check_s3_timestamps:
            modification_dates = {key["LastModified"].day for key in resp["Contents"] if "LastModified" in key}
            if len(modification_dates) != 1:
                raise Exception(
                    "Files in S3 are from muliples dates. This might mean the data is corrupted/duplicated."
                )

        manifest_file_key = next(
            entry["Key"]
            for entry in resp["Contents"]
            if "Key" in entry and entry["Key"].endswith(self._manifest_file_name)
        )

        if use_cache and manifest_file_key:
            self._s3_client.download_file(
                Bucket=self._s3_bucket_name,
                Key=manifest_file_key,
                Filename=str(self._destination_dir / self._manifest_file_name),
            )
            with open(self._destination_dir / self._manifest_file_name) as f:
                manifest: dict[str, Any] = json.load(f)
                if self._is_local_dataset_fresh(manifest):
                    logger.info("Local dataset is up to date with S3 bucket contents, skipping download.")
                    return self._get_local_filenames()

        logger.info(f"Downloading {self._s3_bucket_name}/{self._s3_key_prefix} to {self._destination_dir} from S3.")
        clear_directory(self._destination_dir)
        keys_to_download = [
            entry["Key"]
            for entry in resp["Contents"]
            if "Key" in entry and (entry["Key"].endswith(".csv") or entry["Key"].endswith(".gz"))
        ]

        if keys_to_download:
            self._s3_fetch_all(
                self._destination_dir,
                keys_to_download,
                self._num_workers,
            )
            logger.info(f"Finished downloading {len(keys_to_download)} files from S3.")

        return self._get_local_filenames()

    def _get_local_filenames(self) -> list[Path]:
        """Get a list of local filenames for CSV and GZ files in the destination directory.

        Returns:
            A list of Path objects for all CSV and GZ files in the destination directory.
        """
        return [
            Path(filepath)
            for filepath in glob.glob(f"{self._destination_dir}/*.csv") + glob.glob(f"{self._destination_dir}/*.gz")
        ]

    def _get_s3_client(  # type: ignore
        self,
        aws_profile_name: str,
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

    def _s3_fetch_file(self, key: str, destination: Path) -> None:
        """Downloads a single file from the S3 bucket to the specified destination.

        Args:
            key: S3 key of the file to download.
            destination: Local destination path to store the downloaded file.
        """
        gb = 1024**3
        transfer_config = BotoTransferConfig(
            use_threads=False,
            multipart_threshold=int(0.5 * gb),  # Multipart transfer if file > 500MB
        )
        with open(destination, "wb") as data:
            self._s3_client.download_fileobj(
                Bucket=self._s3_bucket_name,
                Key=key,
                Fileobj=data,
                Config=transfer_config,
            )

    def _s3_fetch_all(
        self,
        directory: Path,
        keys: list[str],
        num_workers: int,
    ) -> None:
        """Downloads all specified files from the S3 bucket to the local directory.

        Args:
            directory: Local destination directory to store the downloaded files.
            keys: List of S3 keys for the files to download.
            num_workers: Number of workers to use for concurrent downloads.
        """
        with tqdm(total=len(keys), desc="Downloading CSV files from S3") as pbar:
            with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for _ in executor.map(
                    self._s3_fetch_file,
                    keys,
                    [(directory / Path(key).name) for key in keys],
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
            local_file_path = self._destination_dir / file_name
            if not local_file_path.exists() or file_size != os.path.getsize(local_file_path):
                return False
        return True
