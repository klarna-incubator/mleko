"""A module for downloading and managing Kaggle datasets using the Kaggle API.

In order to use this module, the user must have valid Kaggle API credentials.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

import requests
from requests.auth import HTTPBasicAuth
from tqdm.auto import tqdm

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.file_helpers import clear_directory

from .base_ingester import BaseIngester


logger = CustomLogger()
"""A module-level custom logger."""


class KaggleCredentials(NamedTuple):
    """Represents a set of Kaggle API credentials, including a username and API key."""

    username: str
    """Username for the Kaggle API."""

    key: str
    """API key for the Kaggle API."""


class KaggleCredentialsManager:
    """Manages retrieval of Kaggle API credentials from environment variables or a file."""

    _ENV_VARIABLE_USERNAME = "KAGGLE_USERNAME"
    """Name of the environment variable containing the Kaggle username."""

    _ENV_VARIABLE_KEY = "KAGGLE_KEY"
    """Name of the environment variable containing the Kaggle API key."""

    _CONFIG_DEFAULT_PATH = "~/.kaggle/kaggle.json"
    """Default path to the Kaggle API credentials file."""

    _CONFIG_VARIABLE_USERNAME = "username"
    """Name of the key in the Kaggle API credentials file containing the Kaggle username."""

    _CONFIG_VARIABLE_KEY = "key"
    """Name of the key in the Kaggle API credentials file containing the Kaggle API key."""

    @staticmethod
    def get_kaggle_credentials(credentials_file_path: str | Path | None = None) -> KaggleCredentials:
        """Retrieves Kaggle API credentials from the specified file, environment variables, or the default location.

        Args:
            credentials_file_path: Path to the Kaggle API credentials file.

        Returns:
            A KaggleCredentials instance with the retrieved username and API key.
        """
        if credentials_file_path is not None:
            logger.info(f"Attempting to fetch Kaggle API credentials from config at {credentials_file_path}.")
            credentials = KaggleCredentialsManager._read_config_file(Path(credentials_file_path))
        else:
            logger.info(
                "Attempting to fetch Kaggle API credentials from environment variables "
                f"{KaggleCredentialsManager._ENV_VARIABLE_USERNAME!r} and "
                f"{KaggleCredentialsManager._ENV_VARIABLE_KEY!r}."
            )
            env_credentials = KaggleCredentialsManager._read_environment_config()

            if env_credentials is None:
                logger.info(
                    "Kaggle API credentials not found in environment variables, attempting to fetch from "
                    f"fallback path at {KaggleCredentialsManager._CONFIG_DEFAULT_PATH}."
                )
                credentials = KaggleCredentialsManager._read_config_file(
                    Path(KaggleCredentialsManager._CONFIG_DEFAULT_PATH)
                )
            else:
                credentials = env_credentials
        logger.info("Kaggle credentials successfully fetched.")

        return credentials

    @staticmethod
    def _read_config_file(credentials_file_path: Path) -> KaggleCredentials:
        """Reads Kaggle API credentials from the given configuration file.

        Args:
            credentials_file_path: Path to the Kaggle API credentials file.

        Returns:
            A KaggleCredentials instance with the retrieved username and API key.

        Raises:
            FileNotFoundError: If the file does not exist or is a directory.
            KeyError: If the username or API key is missing from the credentials JSON.
            JSONDecodeError: If the JSON decoding fails while reading the credentials.
        """
        expanded_credentials_file_path = credentials_file_path.expanduser()
        if not expanded_credentials_file_path.exists():
            error_msg = f"{credentials_file_path} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not expanded_credentials_file_path.is_file():
            error_msg = f"{credentials_file_path} is a directory."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        permissions = expanded_credentials_file_path.stat().st_mode
        if (permissions & 4) or (permissions & 32):
            logger.warning(
                "Your Kaggle API credentials are readable by other users on this system. "
                f"Run `chmod 600 {credentials_file_path}` to fix this issue."
            )

        try:
            with open(expanded_credentials_file_path) as f:
                config_data: dict[str, str] = json.load(f)
                kaggle_config = KaggleCredentials(
                    username=config_data[KaggleCredentialsManager._CONFIG_VARIABLE_USERNAME],
                    key=config_data[KaggleCredentialsManager._CONFIG_VARIABLE_KEY],
                )
                return kaggle_config
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode Kaggle API credentials JSON. {e.msg}: line "
                f"{e.lineno} column {e.colno} (char {e.pos})."
            )
            raise json.JSONDecodeError(e.msg, e.doc, e.pos) from e
        except KeyError as e:
            logger.error(f"Missing {e} from Kaggle API credentials JSON.")
            raise e

    @staticmethod
    def _read_environment_config() -> KaggleCredentials | None:
        """Reads Kaggle API credentials from environment variables.

        Returns:
            A KaggleCredentials instance with the retrieved username and API key, or None
                if the environment variables are not set.
        """
        username = os.getenv(KaggleCredentialsManager._ENV_VARIABLE_USERNAME)
        key = os.getenv(KaggleCredentialsManager._ENV_VARIABLE_KEY)
        return KaggleCredentials(username=username, key=key) if username and key else None


@dataclass
class KaggleFileMetadata:
    """Represents the metadata of a single Kaggle dataset file."""

    name: str
    """Name of the file."""

    creation_timestamp: float
    """Timestamp of the file creation."""

    total_bytes: int
    """Total size of the file in bytes."""


class KaggleIngester(BaseIngester):
    """Handles dataset retrieval from Kaggle, downloading and updating files as necessary.

    The `KaggleIngester` class downloads files from the specified Kaggle dataset and saves them to the destination
    directory. It also checks if the local files are up to date and skips downloading if everything is already in
    place.
    """

    _KAGGLE_API_VERSION = "v1"
    """The Kaggle API version to use."""

    _KAGGLE_DATASET_URL = f"https://www.kaggle.com/api/{_KAGGLE_API_VERSION}/datasets"
    """The base URL for Kaggle dataset API requests."""

    @auto_repr
    def __init__(
        self,
        destination_directory: str | Path,
        owner_slug: str,
        dataset_slug: str,
        file_names: list[str] | None = None,
        dataset_version: str | int | None = None,
        kaggle_api_credentials_file: str | Path | None = None,
        num_workers: int = 64,
    ) -> None:
        """Initializes a `KaggleIngester` instance to fetch data from a specific Kaggle dataset.

        In order to use `KaggleIngester`, valid Kaggle API credentials are required. These credentials can be obtained
        by creating an API token on the Kaggle account settings page. The token should be saved in a JSON file named
        `kaggle.json` containing the "username" and "key" fields.

        There are three possible locations where Kaggle API credentials can be provided:

        1. Custom file location: Pass the file path to `kaggle_api_credentials_file` in the constructor.
        2. Environment variables: Set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.
        3. Default .kaggle folder: Place the `kaggle.json` file into the "~/.kaggle/" directory.

        Note:
            The Kaggle API is not perfect and sometimes returns incorrect metadata for files, where one or more of the
            files are missing from the dataset. This can lead to the wrong files being downloaded or the download
            failing altogether. If you encounter this issue, please report it to Kaggle.

            The issue is observed when the dataset contains a large number of files (e.g. 1000+) or if the dataset
            contains nested folders.

        Args:
            destination_directory: The directory where the downloaded files will be stored.
            owner_slug: The owner's Kaggle username or organization name.
            dataset_slug: The dataset's unique Kaggle identifier (slug).
            file_names: A list of file names to download. If not provided or empty, all files in
                the dataset will be downloaded.
            dataset_version: The specific dataset version number to download. If not provided,
                the latest version will be fetched.
            kaggle_api_credentials_file: Path to a Kaggle API credentials JSON file. If not
                provided, environment variables or the default file location will be used.
            num_workers: Number of concurrent threads to use when downloading files.

        Examples:
            >>> from mleko.dataset.sources import KaggleIngester
            >>> kaggle_ingester = KaggleIngester(
            ...     destination_directory="~/data",
            ...     owner_slug="allen-institute-for-ai",
            ...     dataset_slug="covid-19-masks-dataset",
            ...     file_names=["images.zip", "annotations.json"],
            ...     dataset_version=1,
            ... )
            >>> kaggle_ingester.fetch_data()
            [PosixPath('~/data/images.zip'), PosixPath('~/data/annotations.json')]
        """
        super().__init__(destination_directory)
        self._owner_slug, self._dataset_slug, self._dataset_version = owner_slug, dataset_slug, dataset_version
        self._file_names: set[str] = set(file_names) if file_names is not None else set()
        self._kaggle_config = KaggleCredentialsManager.get_kaggle_credentials(kaggle_api_credentials_file)
        self._num_workers = num_workers

    def fetch_data(self, force_recompute: bool = False) -> list[Path]:
        """Fetches data from the specified Kaggle dataset.

        This method downloads files from the Kaggle dataset and returns the local file paths of the downloaded files.
        The method checks if local files are up-to-date and skips downloading if everything is already in place and
        `force_recompute` is set to False.

        Args:
            force_recompute: If set to False, the method will check if the local files are up-to-date and
                skip downloading if everything is already in place.

        Returns:
            A list of local file paths pointing to the downloaded files.

        Raises:
            ValueError: If Kaggle returns 0 files for the given dataset. This could occur if the API is broken.
        """
        dataset_path = f"{self._owner_slug}/{self._dataset_slug}"

        params: dict[str, str] = {}
        if self._dataset_version:
            params["datasetVersionNumber"] = str(self._dataset_version)

        files_metadata = self._kaggle_fetch_files_metadata(params)

        if len(files_metadata) == 0:
            error_msg = f"Kaggle returned 0 files for the given dataset {dataset_path}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not force_recompute and self._is_local_dataset_fresh(files_metadata):
            logger.info("\033[32mCache Hit\033[0m: Local dataset is up to date with Kaggle, skipping download.")
            return self._get_local_filenames(["gz", "csv", "zip"])

        file_names = "*"
        if self._file_names:
            file_names = f"{list(self._file_names)!r}"

        if force_recompute:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m: Downloading {dataset_path}/{file_names} to "
                f"{self._destination_directory} from Kaggle."
            )
        else:
            logger.info(
                f"\033[31mCache Miss\033[0m: Downloading {dataset_path}/{file_names} to "
                f"{self._destination_directory} from Kaggle."
            )

        clear_directory(self._destination_directory)

        kaggle_file_paths = [f"{dataset_path}/{file_metadata.name}" for file_metadata in files_metadata]

        if kaggle_file_paths:
            self._kaggle_fetch_files(kaggle_file_paths, params)
            logger.info(f"Finished downloading {len(kaggle_file_paths)} files from Kaggle.")

        return self._get_local_filenames(["gz", "csv", "zip"])

    def _kaggle_fetch_files_metadata(self, params: dict[str, str]) -> list[KaggleFileMetadata]:
        """Fetch the metadata of the files in the dataset.

        When fetching the metadata, the API returns a list of files in the dataset. The list contains the name of the
        file, the creation date, and the file size.

        Args:
            params: A dictionary of query parameters to pass to the Kaggle API.

        Raises:
            HTTPError: If there is an error in the HTTP response while requesting file list from Kaggle.

        Returns:
            A list of KaggleFileMetadata objects containing the metadata of the files in the dataset.
        """
        list_files_response = requests.get(
            f"{self._KAGGLE_DATASET_URL}/list/{self._owner_slug}/{self._dataset_slug}",
            params=params,
            auth=HTTPBasicAuth(self._kaggle_config.username, self._kaggle_config.key),
            timeout=5,
        )

        try:
            list_files_response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(e)
            raise requests.HTTPError(e) from e

        files_metadata: list[KaggleFileMetadata] = [
            KaggleFileMetadata(
                file_metadata["name"],
                datetime.strptime(file_metadata["creationDate"].split(".")[0], "%Y-%m-%dT%H:%M:%S").timestamp(),
                int(file_metadata["totalBytes"]),
            )
            for file_metadata in json.loads(list_files_response.content)["datasetFiles"]
            if file_metadata["name"] in self._file_names or len(self._file_names) == 0
        ]
        return files_metadata

    def _kaggle_fetch_file(self, kaggle_file_path: str, params: dict[str, str]) -> None:
        """Downloads a single Kaggle dataset file and saves it in the destination directory.

        Args:
            kaggle_file_path: The Kaggle file path to download.
            params: The request parameters containing the dataset version number, if applicable.

        Raises:
            HTTPError: If there is an error in the HTTP response while downloading file from Kaggle.
        """
        local_file_path = self._destination_directory / Path(kaggle_file_path).name
        response = requests.get(
            f"{self._KAGGLE_DATASET_URL}/download/{kaggle_file_path}",
            params=params,
            auth=HTTPBasicAuth(self._kaggle_config.username, self._kaggle_config.key),
            timeout=5,
            stream=True,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(e)
            raise requests.HTTPError(e) from e

        with open(local_file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        file_signature = None
        with open(local_file_path, "rb") as f:
            file_signature = f.read(4)

        # Check if file is a zip file and unzip it if it is.
        if file_signature == b"PK\x03\x04":
            local_file_path = local_file_path.rename(local_file_path.with_name(local_file_path.name + ".zip"))
            shutil.unpack_archive(local_file_path, local_file_path.parent)
            local_file_path.unlink()

    def _kaggle_fetch_files(self, kaggle_file_paths: list[str], params: dict[str, str]) -> None:
        """Downloads multiple Kaggle dataset files concurrently.

        Args:
            kaggle_file_paths: A list of Kaggle file paths to download.
            params: The request parameters containing the dataset version number, if applicable.
        """
        with tqdm(total=len(kaggle_file_paths), desc="Downloading files from Kaggle") as pbar:
            with futures.ThreadPoolExecutor(max_workers=min(self._num_workers, len(kaggle_file_paths))) as executor:
                for _ in executor.map(
                    self._kaggle_fetch_file,
                    kaggle_file_paths,
                    repeat(params),
                ):
                    pbar.update(1)

    def _is_local_dataset_fresh(self, files_metadata: list[KaggleFileMetadata]) -> bool:
        """Checks if the local dataset files are up to date with the Kaggle dataset files.

        Comparing file size and modification timestamp, this method determines if the local files are up to date and
        if they match the remote Kaggle dataset files.

        Args:
            files_metadata: A list containing the metadata of the files in the Kaggle dataset.

        Returns:
            True if the local dataset files are up to date, False otherwise.
        """
        for file in files_metadata:
            local_file_path = self._destination_directory / file.name
            if (
                not local_file_path.exists()
                or file.total_bytes != os.path.getsize(local_file_path)
                or file.creation_timestamp > os.path.getmtime(local_file_path)
            ):
                return False
        return True
