"""Test suite for the `dataset.data_source.kaggle_data_source` module."""
from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from pytest import fixture, raises

from mleko.dataset.data_source.kaggle_data_source import (
    KaggleCredentials,
    KaggleCredentialsManager,
    KaggleDataSource,
    KaggleFileMetadata,
)


class TestKaggleCredentials:
    """Test suite for `dataset.data_source.kaggle_data_source.KaggleCredentials`."""

    def test_init(self):
        """Should successfully initialize."""
        username = "dummy_username"
        key = "dummy_key"
        kaggle_credentials = KaggleCredentials(username=username, key=key)

        assert kaggle_credentials.username == username
        assert kaggle_credentials.key == key


class TestCredentialsManager:
    """Test suite for `dataset.data_source.kaggle_data_source.TestCredentialsManager`."""

    def test_get_credentials_from_config_file(self, temporary_directory: Path):
        """Should fetch credentials from a provided config file."""
        credentials = KaggleCredentials(username="dummy_username", key="dummy_key")
        config_path = temporary_directory / "kaggle.json"
        with open(config_path, "w") as f:
            json.dump({"username": credentials.username, "key": credentials.key}, f)

        fetched_credentials = KaggleCredentialsManager.get_kaggle_credentials(credentials_file_path=config_path)
        assert fetched_credentials == credentials

    def test_get_credentials_from_environment_variables(self):
        """Should fetch credentials from environment variables."""
        credentials = KaggleCredentials(username="dummy_username", key="dummy_key")
        with patch.dict(os.environ, {"KAGGLE_USERNAME": credentials.username, "KAGGLE_KEY": credentials.key}):
            fetched_credentials = KaggleCredentialsManager.get_kaggle_credentials()
            assert fetched_credentials == credentials

    def test_fallback_to_default_path(self):
        """Should falls back to default directory when no credentials are found elsewhere."""
        credentials = KaggleCredentials(username="dummy_username", key="dummy_key")

        with patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
            with patch(
                "mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager._read_environment_config"
            ) as mock_read_env_config, patch(
                "mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager._read_config_file"
            ) as mock_read_config_file:
                mock_read_env_config.return_value = None
                mock_read_config_file.return_value = credentials

                fetched_credentials = KaggleCredentialsManager.get_kaggle_credentials()

                default_path = Path(KaggleCredentialsManager._CONFIG_DEFAULT_PATH)
                mock_read_config_file.assert_called_with(default_path)

                assert fetched_credentials == credentials

    def test_file_not_found_error(self, temporary_directory: Path):
        """Should raise FileNotFoundError if the config file is not found."""
        config_path = temporary_directory / "non_existent_file.json"
        with raises(FileNotFoundError):
            KaggleCredentialsManager.get_kaggle_credentials(credentials_file_path=config_path)

    def test_given_directory_raises_error(self, temporary_directory: Path):
        """Should raise FileNotFoundError if a directory is provided."""
        with raises(FileNotFoundError):
            KaggleCredentialsManager.get_kaggle_credentials(credentials_file_path=temporary_directory)

    def test_json_decode_error(self, temporary_directory: Path):
        """Should raise JSONDecodeError if the JSON file is not properly formatted."""
        config_path = temporary_directory / "malformed_kaggle.json"
        with open(config_path, "w") as f:
            f.write("{ MALFORMED JSON}")

        with raises(json.JSONDecodeError):
            KaggleCredentialsManager.get_kaggle_credentials(credentials_file_path=config_path)

    def test_key_error(self, temporary_directory: Path):
        """Should raise KeyError if a required key is missing from the JSON file."""
        config_path = temporary_directory / "incomplete_kaggle.json"
        with open(config_path, "w") as f:
            json.dump({"username": "dummy_username"}, f)

        with raises(KeyError):
            KaggleCredentialsManager.get_kaggle_credentials(credentials_file_path=config_path)


class TestKaggleFileMetadata:
    """Test suite for `dataset.data_source.kaggle_data_source.KaggleFileMetadata`."""

    def test_init(self):
        """Should successfully initialize."""
        name = "sample.csv"
        creation_timestamp = 1577836799.0
        total_bytes = 12345
        kaggle_file_metadata = KaggleFileMetadata(name, creation_timestamp, total_bytes)

        assert kaggle_file_metadata.name == name
        assert kaggle_file_metadata.creation_timestamp == creation_timestamp
        assert kaggle_file_metadata.total_bytes == total_bytes


class TestKaggleDataSource:
    """Test suite for `dataset.data_source.kaggle_data_source.KaggleDataSource`."""

    @fixture
    def sample_kaggle_credentials(self):
        """Returns a sample `KaggleCredentials` instance."""
        return KaggleCredentials(username="dummy_username", key="dummy_key")

    @fixture
    def sample_kaggle_file_metadata(self):
        """Returns a sample `KaggleFileMetadata` instance."""
        return KaggleFileMetadata("file.csv", 1609459200.0, 12345)

    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    def test_init(self, mock_get_credentials: MagicMock, sample_kaggle_credentials: KaggleCredentials):
        """Should successfullt initialize."""
        mock_get_credentials.return_value = sample_kaggle_credentials
        data_source = KaggleDataSource(
            destination_directory=".",
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
        )

        assert isinstance(data_source, KaggleDataSource)
        assert data_source._kaggle_config == sample_kaggle_credentials

    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleDataSource._kaggle_fetch_files")
    def test_fetch_data_when_cache_is_fresh(
        self,
        mock_kaggle_fetch_files: MagicMock,
        mock_get_credentials: MagicMock,
        sample_kaggle_credentials: KaggleCredentials,
        temporary_directory: Path,
    ):
        """Should skip download the local cache is fresh."""
        mock_get_credentials.return_value = sample_kaggle_credentials

        files = [
            {"name": "file1.csv", "creationDate": "2020-01-01T00:00:00.000Z", "totalBytes": 50},
            {"name": "file2.csv", "creationDate": "2020-02-01T00:00:00.000Z", "totalBytes": 75},
        ]
        for file in files:
            fname = file["name"]
            fdate = datetime.strptime(file["creationDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
            fsize = file["totalBytes"]

            with open(temporary_directory / fname, "w") as file:
                file.seek(fsize - 1)
                file.write("\0")

            os.utime(temporary_directory / fname, (fdate.timestamp(), fdate.timestamp()))

        data_source = KaggleDataSource(
            destination_directory=temporary_directory,
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
            dataset_version="dummy_version",
        )

        with patch.object(requests, "get") as mock_requests_get:
            mock_requests_get.return_value = MagicMock(
                status_code=200,
                content=json.dumps({"datasetFiles": files}),
            )
            files = data_source.fetch_data()

        assert len(files) == 2
        assert files == [temporary_directory / "file1.csv", temporary_directory / "file2.csv"]
        mock_kaggle_fetch_files.assert_not_called()

    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    def test_raise_http_error_fetch_dataset_list(
        self,
        mock_get_credentials: MagicMock,
        sample_kaggle_credentials: KaggleCredentials,
        temporary_directory: Path,
    ):
        """Should raise an HTTPError if the dataset list fails."""
        mock_get_credentials.return_value = sample_kaggle_credentials

        data_source = KaggleDataSource(
            destination_directory=temporary_directory,
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
            dataset_version="dummy_version",
        )

        with patch.object(requests, "get") as mock_requests_get:
            mock_requests_get.return_value = MagicMock(
                status_code=401,
                reason="Unauthorized",
                raise_for_status=MagicMock(side_effect=requests.exceptions.HTTPError),
            )
            with raises(requests.exceptions.HTTPError):
                data_source.fetch_data()

    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    def test_raise_value_error_if_dataset_list_is_empty(
        self,
        mock_get_credentials: MagicMock,
        sample_kaggle_credentials: KaggleCredentials,
        temporary_directory: Path,
    ):
        """Should raise a ValueError if the dataset list returns an empty array."""
        mock_get_credentials.return_value = sample_kaggle_credentials

        data_source = KaggleDataSource(
            destination_directory=temporary_directory,
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
            dataset_version="dummy_version",
        )

        with patch.object(requests, "get") as mock_requests_get:
            mock_requests_get.return_value = MagicMock(
                requests.Response,
                status_code=200,
                content=json.dumps({"datasetFiles": []}),
            )
            with raises(ValueError):
                data_source.fetch_data()

    @pytest.mark.parametrize("force_recompute", [True, False])
    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    @patch("shutil.unpack_archive")
    def test_fetch_data_when_cache_is_outdated(
        self,
        mock_get_credentials: MagicMock,
        mock_unpack_archive: MagicMock,
        sample_kaggle_credentials: KaggleCredentials,
        temporary_directory: Path,
        force_recompute: bool,
    ):
        """Should download the files if local ones are outdated."""
        mock_get_credentials.return_value = sample_kaggle_credentials

        files = [
            {"name": "file1.csv", "creationDate": "2020-01-01T00:00:00.000Z", "totalBytes": 50},
            {"name": "file2.csv", "creationDate": "2020-02-01T00:00:00.000Z", "totalBytes": 75},
        ]
        for file in files:
            fname = file["name"]
            fdate = datetime.strptime(file["creationDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
            fsize = file["totalBytes"] + 1  # Make files different in size

            with open(temporary_directory / fname, "w") as file:
                file.seek(fsize - 1)
                file.write("\0")

            os.utime(temporary_directory / fname, (fdate.timestamp(), fdate.timestamp()))

        data_source = KaggleDataSource(
            destination_directory=temporary_directory,
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
            dataset_version="dummy_version",
            file_names=["file1.csv", "file2.csv"],
        )

        with patch.object(requests, "get") as mock_requests_get:
            file_content = b"file1,file2\n1,2\n"
            zip_file_content = b"PK\x03\x04,file2\n1,2\n"  # signature of a zip file
            mock_requests_get.side_effect = [
                MagicMock(
                    status_code=200,
                    content=json.dumps({"datasetFiles": files}),
                ),
                MagicMock(
                    requests.Response,
                    status_code=200,
                    stream=True,
                    raw=io.BytesIO(file_content),
                    iter_content=MagicMock(return_value=iter([file_content])),
                ),
                MagicMock(
                    requests.Response,
                    status_code=200,
                    stream=True,
                    raw=io.BytesIO(zip_file_content),
                    iter_content=MagicMock(return_value=iter([zip_file_content])),
                ),
            ]
            files = data_source.fetch_data(force_recompute=force_recompute)

        assert len(files) == 1
        assert files == [temporary_directory / "file1.csv"]  # the zip file should be ignored
        assert mock_unpack_archive.call_count == 1

    @patch("mleko.dataset.data_source.kaggle_data_source.KaggleCredentialsManager.get_kaggle_credentials")
    def test_fetch_data_raises_http_error_on_bad_request(
        self,
        mock_get_credentials: MagicMock,
        sample_kaggle_credentials: KaggleCredentials,
        temporary_directory: Path,
    ):
        """Should raise HTTPError if some requested file fails to download."""
        mock_get_credentials.return_value = sample_kaggle_credentials

        files = [
            {"name": "file1.csv", "creationDate": "2020-01-01T00:00:00.000Z", "totalBytes": 50},
            {"name": "file2.csv", "creationDate": "2020-02-01T00:00:00.000Z", "totalBytes": 75},
        ]
        for file in files:
            fname = file["name"]
            fdate = datetime.strptime(file["creationDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
            fsize = file["totalBytes"] + 1  # Make files different in size

            with open(temporary_directory / fname, "w") as file:
                file.seek(fsize - 1)
                file.write("\0")

            os.utime(temporary_directory / fname, (fdate.timestamp(), fdate.timestamp()))

        data_source = KaggleDataSource(
            destination_directory=temporary_directory,
            owner_slug="dummy_owner",
            dataset_slug="dummy_dataset",
            dataset_version="dummy_version",
            file_names=["file1.csv", "file2.csv"],
        )

        with patch.object(requests, "get") as mock_requests_get:
            file_content = b"file1,file2\n1,2\n"
            mock_requests_get.side_effect = [
                MagicMock(
                    status_code=200,
                    content=json.dumps({"datasetFiles": files}),
                ),
                MagicMock(
                    requests.Response,
                    status_code=200,
                    stream=True,
                    raw=io.BytesIO(file_content),
                    iter_content=MagicMock(return_value=iter([file_content])),
                ),
                MagicMock(
                    requests.Response,
                    status_code=401,
                    reason="Unauthorized",
                    raise_for_status=MagicMock(side_effect=requests.exceptions.HTTPError),
                ),
            ]

            with raises(requests.exceptions.HTTPError):
                data_source.fetch_data()
