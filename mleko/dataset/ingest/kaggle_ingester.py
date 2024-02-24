"""A module for downloading and managing Kaggle datasets using the Kaggle API.

In order to use this module, the user must have valid Kaggle API credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

import requests
from requests.auth import HTTPBasicAuth
from tqdm.auto import tqdm

from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr

from .base_ingester import BaseIngester, LocalFileEntry, LocalManifestHandler


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
            key_error: If the username or API key is missing from the credentials JSON.
            JSONDecodeError: If the JSON decoding fails while reading the credentials.
        """
        expanded_credentials_file_path = credentials_file_path.expanduser()
        if not expanded_credentials_file_path.exists():
            msg = f"{credentials_file_path} does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not expanded_credentials_file_path.is_file():
            msg = f"{credentials_file_path} is a directory."
            logger.error(msg)
            raise FileNotFoundError(msg)

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
        except KeyError as key_error:
            logger.error(f"Missing {key_error} from Kaggle API credentials JSON.")
            raise key_error

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
class KaggleFileManifest:
    """Manifest entry for a single file in a Kaggle dataset."""

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
        owner_slug: str,
        dataset_slug: str,
        file_pattern: str | list[str] = "*",
        dataset_id: str | None = None,
        cache_directory: str | Path = "data/kaggle-ingester",
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
            owner_slug: The owner's Kaggle username or organization name.
            dataset_slug: The dataset's unique Kaggle identifier (slug).
            file_pattern: Pattern to match the files to download, e.g. `*.csv` or [`*.csv`, `*.json`], etc.
                For more information, see https://docs.python.org/3/library/fnmatch.html.
            dataset_id: Id of the dataset to be used instead of the default fingerprint (MD5 hash of the owner slug,
                dataset version, and dataset slug). Note that this will overwrite any existing dataset with the same
                name in the destination directory, so make sure to use a unique name.
            cache_directory: The directory where the downloaded files will be stored.
            dataset_version: The specific dataset version number to download. If not provided,
                the latest version will be fetched.
            kaggle_api_credentials_file: Path to a Kaggle API credentials JSON file. If not
                provided, environment variables or the default file location will be used.
            num_workers: Number of concurrent threads to use when downloading files.

        Examples:
            >>> from mleko.dataset.sources import KaggleIngester
            >>> kaggle_ingester = KaggleIngester(
            ...     owner_slug="allen-institute-for-ai",
            ...     dataset_slug="covid-19-masks-dataset",
            ...     file_pattern="file_*.zip",
            ...     dataset_id="covid-19", # Optional, but will store the data in "./data/covid-19/" instead of
            ...                            # "./data/<fingerprint>/".
            ...     dataset_version=1,
            ... )
            >>> kaggle_ingester.fetch_data()
            [PosixPath('~/data/covid-19/file_1.zip'), PosixPath('~/data/covid-19/file_2.zip')]
        """
        dataset_id = (
            dataset_id
            if dataset_id is not None
            else hashlib.md5((owner_slug + dataset_slug + str(dataset_version)).encode()).hexdigest()
        )
        super().__init__(cache_directory, dataset_id)
        self._local_manifest_handler = LocalManifestHandler(
            self._cache_directory / f"{self._fingerprint}.manifest.json"
        )
        self._owner_slug = owner_slug
        self._dataset_slug = dataset_slug
        self._dataset_version = dataset_version
        self._kaggle_config = KaggleCredentialsManager.get_kaggle_credentials(kaggle_api_credentials_file)
        self._num_workers = num_workers

        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]
        self._file_pattern = file_pattern

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
        """
        dataset_path = f"{self._owner_slug}/{self._dataset_slug}"

        params: dict[str, str] = {}
        if self._dataset_version:
            params["datasetVersionNumber"] = str(self._dataset_version)
        kaggle_manifest = self._build_kaggle_manifest(params)

        if force_recompute:
            logger.info(
                f"\033[33mForce Cache Refresh\033[0m: Downloading files matching {self._file_pattern} from "
                f"{dataset_path} to {self._cache_directory} from Kaggle."
            )
        else:
            if self._is_local_dataset_fresh(kaggle_manifest):
                logger.info("\033[32mCache Hit\033[0m: Local dataset is up to date with Kaggle, skipping download.")
                local_file_names = set(self._local_manifest_handler.get_file_names())
                kaggle_file_names: set[str] = {kaggle_file.name for kaggle_file in kaggle_manifest}
                files_to_delete = list(local_file_names.difference(kaggle_file_names))

                if len(files_to_delete) > 0:
                    logger.info(
                        f"Deleting {len(files_to_delete)} files from "
                        f"{self._cache_directory} that are no longer present in Kaggle or filtered out."
                    )

                self._delete_local_files(files_to_delete)
                self._local_manifest_handler.remove_files(files_to_delete)
                return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

            logger.info(
                f"\033[31mCache Miss\033[0m: Downloading files matching {self._file_pattern} from "
                f"{dataset_path} to {self._cache_directory} from Kaggle."
            )

        self._delete_local_files(self._local_manifest_handler.get_file_names())
        kaggle_file_paths = [f"{dataset_path}/{file_metadata.name}" for file_metadata in kaggle_manifest]
        if len(kaggle_file_paths) > 0:
            self._kaggle_fetch_files(kaggle_file_paths, params)
            self._local_manifest_handler.set_files(
                [
                    LocalFileEntry(
                        name=Path(kaggle_file_paths[0]).name,
                        size=os.path.getsize(self._cache_directory / Path(kaggle_file_paths[0]).name),
                    )
                ]
            )
            logger.info(f"Finished downloading {len(kaggle_file_paths)} files from Kaggle.")

        return self._get_full_file_paths(self._local_manifest_handler.get_file_names())

    def _build_kaggle_manifest(self, params: dict[str, str]) -> list[KaggleFileManifest]:
        """Fetch the metadata of the files in the dataset.

        When fetching the metadata, the API returns a list of files in the dataset. The list contains the name of the
        file, the creation date, and the file size.

        Args:
            params: A dictionary of query parameters to pass to the Kaggle API.

        Raises:
            HTTPError: If there is an error in the HTTP response while requesting file list from Kaggle.

        Returns:
            A list of KaggleFileManifest objects containing the metadata of the files in the dataset.
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

        kaggle_manifest: list[KaggleFileManifest] = [
            KaggleFileManifest(
                file["name"],
                datetime.strptime(file["creationDate"].split(".")[0], "%Y-%m-%dT%H:%M:%S").timestamp(),
                int(file["totalBytes"]),
            )
            for file in json.loads(list_files_response.content)["datasetFiles"]
            if any(fnmatch(file["name"], pattern) for pattern in self._file_pattern)
        ]

        if len(kaggle_manifest) == 0:
            msg = (
                f"No files matching {self._file_pattern} found in Kaggle dataset "
                f"{self._owner_slug}/{self._dataset_slug}."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Found {len(kaggle_manifest)} file(s) matching any of {self._file_pattern} in Kaggle dataset.")

        return kaggle_manifest

    def _kaggle_fetch_file(self, kaggle_file_path: str, params: dict[str, str]) -> None:
        """Downloads a single Kaggle dataset file and saves it in the destination directory.

        Args:
            kaggle_file_path: The Kaggle file path to download.
            params: The request parameters containing the dataset version number, if applicable.

        Raises:
            HTTPError: If there is an error in the HTTP response while downloading file from Kaggle.
        """
        local_file_path = self._cache_directory / Path(kaggle_file_path).name
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

    def _is_local_dataset_fresh(self, files_metadata: list[KaggleFileManifest]) -> bool:
        """Checks if the local dataset files are up to date with the Kaggle dataset files.

        Comparing file size and modification timestamp, this method determines if the local files are up to date and
        if they match the remote Kaggle dataset files.

        Args:
            files_metadata: A list containing the metadata of the files in the Kaggle dataset.

        Returns:
            True if the local dataset files are up to date, False otherwise.
        """
        for file in files_metadata:
            local_file_path = self._cache_directory / file.name
            if (
                not local_file_path.exists()
                or file.total_bytes != os.path.getsize(local_file_path)
                or file.creation_timestamp > os.path.getmtime(local_file_path)
            ):
                return False
        return True
