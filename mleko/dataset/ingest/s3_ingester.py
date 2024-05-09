"""Module for fetching data from AWS S3 and storing it locally using the `S3Ingester` class."""

from __future__ import annotations

import hashlib
import os
from concurrent import futures
from itertools import repeat
from pathlib import Path

from tqdm.auto import tqdm

from mleko.utils import CustomLogger, LocalFileEntry, LocalManifestHandler, S3Client, auto_repr

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
        s3_bucket_name: str,
        s3_key_prefix: str,
        file_pattern: str | list[str] = "*",
        dataset_id: str | None = None,
        destination_directory: str | Path = "data/s3-ingester",
        aws_profile_name: str | None = None,
        aws_region_name: str = "eu-west-1",
        max_concurrent_files: int = 64,
        workers_per_file: int = 1,
        manifest_file_name: str | None = "manifest",
        s3_timestamp_tolerance: int = -1,
    ) -> None:
        """Initializes the S3 bucket client, configures the cache directory, and sets client-related parameters.

        Note:
            The S3 bucket client is initialized using the provided AWS profile and region. If no profile is provided,
            the default profile will be used. If no region is provided, the default region will be used.

            The profile and region is read from the AWS credentials file located at '~/.aws/credentials'.

        Warning:
            The `max_concurrent_files` and `workers_per_file` parameters are used to control the
            number of concurrent downloads and parts downloaded per file, respectively. These parameters should be
            set based on the available system resources and the S3 bucket's performance limits. The total number of
            concurrent threads is the product of these two parameters
            (i.e., `max_concurrent_files * workers_per_file`).

        Args:
            s3_bucket_name: Name of the S3 bucket containing the data.
            s3_key_prefix: Prefix of the S3 keys for the files to download
            file_pattern: Pattern to match the files to download, e.g. `*.csv` or [`*.csv`, `*.json`], etc.
                For more information, see https://docs.python.org/3/library/fnmatch.html.
            dataset_id: Id of the dataset to be used instead of the default fingerprint (MD5 hash of the bucket
                name, key prefix, and region name). Note that this will overwrite any existing dataset with the same
                name in the cache directory, so make sure to use a unique name.
            destination_directory: Directory to store the fetched data locally.
            aws_profile_name: AWS profile name to use.
            aws_region_name: AWS region name where the S3 bucket is located.
            max_concurrent_files: Maximum number of files to download concurrently.
            workers_per_file: Number of parts to download concurrently for each file. This is useful for
                downloading large files faster, as it allows for parallel downloads of different parts of the file.
            manifest_file_name: Name of the manifest file located on S3. If provided, the manifest from S3 will
                be used to determine the files to include, before applying the file pattern.
            s3_timestamp_tolerance: Tolerance in hours for the difference in last modified timestamps of files in the S3
                bucket. If the difference is greater than this value, an exception will be raised. If set to -1, no
                check will be performed.

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
            ...     s3_timestamp_tolerance=2,
            ... )
            >>> s3_ingester.fetch_data()
            [PosixPath('data/indian_food/indian_food.csv')]
        """
        dataset_id = (
            dataset_id
            if dataset_id is not None
            else hashlib.md5((s3_bucket_name + s3_key_prefix + aws_region_name).encode()).hexdigest()
        )
        super().__init__(destination_directory, dataset_id)
        self._local_manifest_handler = LocalManifestHandler(
            self._destination_directory / f"{self._fingerprint}.manifest.json"
        )
        self._s3_bucket_name = s3_bucket_name
        self._s3_key_prefix = s3_key_prefix
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._s3_client = S3Client(self._aws_profile_name, self._aws_region_name)
        self._max_concurrent_files = max_concurrent_files
        self._workers_per_file = workers_per_file
        self._manifest_file_name = manifest_file_name
        self._s3_timestamp_tolerance = s3_timestamp_tolerance

        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]
        self._file_pattern = file_pattern

    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Downloads the data from the S3 bucket and stores it in the 'destination_directory'.

        If 'force_recompute' is False, verifies whether the data in the local 'destination_directory' is current
        with the S3 bucket contents based on the manifest file, and skips downloading if it is up to date.

        Args:
            force_recompute: Whether to force the data source to recompute its output, even if it already exists.

        Raises:
            Exception: If files in the S3 bucket have different last modified dates, indicating potential corruption
                       or duplication.
            FileNotFoundError: If no files matching the file pattern are found in the S3 bucket.

        Returns:
            A list of Path objects pointing to the downloaded data files.
        """
        self._s3_client.refresh_client()
        s3_manifest = self._s3_client.get_s3_manifest(
            bucket_name=self._s3_bucket_name,
            key_prefix=self._s3_key_prefix,
            manifest_file_name=self._manifest_file_name,
            file_pattern=self._file_pattern,
        )
        if len(s3_manifest) == 0:
            msg = (
                f"No files matching {self._file_pattern} found in S3 bucket "
                f"{self._s3_bucket_name}/{self._s3_key_prefix}."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Found {len(s3_manifest)} file(s) matching any of {self._file_pattern} in S3 bucket.")

        s3_path_string = f"s3://{Path(self._s3_bucket_name) / self._s3_key_prefix}/"
        if self._s3_timestamp_tolerance >= 0:
            modification_datetimes = [key.last_modified for key in s3_manifest]
            diff_modification_datetimes = max(modification_datetimes) - min(modification_datetimes)
            if diff_modification_datetimes.total_seconds() > self._s3_timestamp_tolerance * 3600:
                error_msg = (
                    f"Files in S3 bucket {s3_path_string} have last modified "
                    f"timestamps differing by more than {self._s3_timestamp_tolerance} hours. "
                    f"Potential corruption or duplication detected."
                )
                logger.error(error_msg)
                raise Exception(error_msg)

        if force_recompute:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m: Downloading files matching {self._file_pattern} from "
                f"{s3_path_string} to {self._destination_directory}."
            )
        else:
            if self._s3_client.is_local_dataset_up_to_date(self._destination_directory, s3_manifest):
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
                        f"{self._destination_directory} that are no longer present in S3 or filtered out."
                    )

                self._delete_local_files(files_to_delete)
                self._local_manifest_handler.remove_files(files_to_delete)
                return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

            logger.info(
                f"\033[31mCache Miss\033[0m: Downloading files matching {self._file_pattern} from "
                f"{s3_path_string} to {self._destination_directory}."
            )

        self._delete_local_files(self._local_manifest_handler.get_file_names())
        keys_to_download: list[str] = [str(s3_file.key) for s3_file in s3_manifest]
        if len(keys_to_download) > 0:
            self._s3_fetch_all(keys_to_download)
            self._local_manifest_handler.set_files(
                [
                    LocalFileEntry(
                        name=Path(key).name, size=os.path.getsize(self._destination_directory / Path(key).name)
                    )
                    for key in keys_to_download
                ]
            )
            logger.info(f"Finished downloading {len(keys_to_download)} files from S3.")

        return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

    def _s3_fetch_all(self, keys: list[str]) -> None:
        """Downloads all specified files from the S3 bucket to the local directory concurrently.

        Args:
            keys: List of S3 keys for the files to download.
        """
        with tqdm(total=len(keys), desc="Downloading files from S3") as pbar:
            with futures.ThreadPoolExecutor(max_workers=min(len(keys), self._max_concurrent_files)) as executor:
                for _ in executor.map(
                    self._s3_client.download_file,
                    repeat(self._destination_directory),
                    repeat(self._s3_bucket_name),
                    keys,
                    repeat(self._workers_per_file),
                ):
                    pbar.update(1)
