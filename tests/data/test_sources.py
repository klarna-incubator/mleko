"""Test suite for the `data.sources` module."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import boto3
import moto
import pytest
from botocore.exceptions import ProfileNotFound
from moto.s3.models import s3_backends
from mypy_boto3_s3.service_resource import Bucket

from mleko.data.sources import BaseDataSource, S3DataSource


class TestBaseDataSource:
    """Test suite for `data.sources.BaseDataSource`."""

    class DataSource(BaseDataSource):
        """Test class inheriting from the `BaseDataSource`."""

        def fetch_data(self, _use_cache):
            """Fetch data."""
            pass

    def test_init(self, temporary_directory: Path):
        """Should create the destination directory and sets _destination_dir attribute."""
        test_data = self.DataSource(temporary_directory)
        assert temporary_directory.exists()
        assert test_data._destination_dir == temporary_directory


class TestS3DataSource:
    """Test suite for `data.sources.S3DataSource`."""

    @pytest.fixture(scope="class")
    def s3_bucket(self) -> Generator[Bucket, None, None]:
        """Mock S3 Bucket."""
        with moto.mock_s3():
            s3 = boto3.resource("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            yield s3.Bucket("test-bucket")

    def test_download(self, s3_bucket: Bucket, temporary_directory: Path):
        """Should download data to temp dir."""
        s3_bucket.Object("test-prefix/manifest").put(Body='{"entries": []}')
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="test-file-data")

        test_data = S3DataSource(
            destination_dir=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=False,
        )
        test_data.fetch_data(
            use_cache=False,
        )

        assert (temporary_directory / "test-file.csv").read_text() == "test-file-data"

    def test_different_timestamps(self, s3_bucket: Bucket, temporary_directory: Path):
        """Should throw exception due to different timestamps of files."""
        s3_bucket.Object("test-prefix/manifest").put(Body='{"entries": []}')
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="test-file-data")
        s3_backends["123456789012"]["global"].buckets["test-bucket"].keys[
            "test-prefix/test-file.csv"
        ].last_modified = datetime.datetime(2000, 1, 1)

        with pytest.raises(Exception, match="Files in S3"):
            test_data = S3DataSource(
                destination_dir=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                num_workers=1,
                check_s3_timestamps=True,
            )
            test_data.fetch_data(
                use_cache=False,
            )

    def test_is_cached(self, s3_bucket: Bucket, temporary_directory: Path):
        """Should use cached files if manifest matches with local files in temp dir."""
        s3_bucket.Object("test-prefix/manifest").put(
            Body=json.dumps(
                {
                    "entries": [
                        {"url": f"{temporary_directory}/test-prefix/test-file.csv", "meta": {"content_length": 5}}
                    ]
                }
            )
        )
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="MLEKO")
        test_data = S3DataSource(
            destination_dir=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        test_data.fetch_data(
            use_cache=False,
        )

        with patch.object(S3DataSource, "_s3_fetch_all") as mocked_s3_fetch_all:
            test_data = S3DataSource(
                destination_dir=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                num_workers=1,
                check_s3_timestamps=False,
            )
            test_data.fetch_data(
                use_cache=True,
            )
            mocked_s3_fetch_all.assert_not_called()

    def test_is_outdated_cache(self, s3_bucket: Bucket, temporary_directory: Path):
        """Should not use cached files if manifest does not matche with local files in temp dir."""
        s3_bucket.Object("test-prefix/manifest").put(
            Body=json.dumps(
                {
                    "entries": [
                        {"url": f"{temporary_directory}/test-prefix/test-file.csv", "meta": {"content_length": 5}}
                    ]
                }
            )
        )
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="MLEKO")
        test_data = S3DataSource(
            destination_dir=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        test_data.fetch_data(
            use_cache=False,
        )

        s3_bucket.Object("test-prefix/manifest").put(
            Body=json.dumps(
                {
                    "entries": [
                        {"url": f"{temporary_directory}/test-prefix/test-file.csv", "meta": {"content_length": 9}}
                    ]
                }
            )
        )
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="MLEKO_NEW")
        with patch.object(S3DataSource, "_s3_fetch_all") as mocked_s3_fetch_all:
            test_data = S3DataSource(
                destination_dir=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                num_workers=1,
                check_s3_timestamps=False,
            )
            test_data.fetch_data(
                use_cache=True,
            )
            mocked_s3_fetch_all.assert_called()

    def test_custom_aws_profile_and_region_name(self, temporary_directory: Path):
        """Should init with custom aws_profile_name and aws_region_name."""
        with patch("boto3.Session.__init__") as mocked_session_init, patch(
            "boto3.Session.get_credentials"
        ) as mocked_get_credentials, patch("boto3.client") as mocked_client:
            # Return None for the default region_name and profile_name
            def side_effect(*args, **kwargs):
                if kwargs.get("region_name") == "us-west-2" and kwargs.get("profile_name") == "custom-profile-name":
                    kwargs["session"] = MagicMock()
                    return None
                raise ProfileNotFound(profile=kwargs.get("profile_name"))  # type: ignore

            mocked_session_init.side_effect = side_effect

            credentials = MagicMock()
            credentials.access_key = "test_key"
            credentials.secret_key = "test_secret"  # noqa: S105
            credentials.token = "test_token"  # noqa: S105
            mocked_get_credentials.return_value = credentials

            S3DataSource(
                destination_dir=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_profile_name="custom-profile-name",
                aws_region_name="us-west-2",
                num_workers=1,
                check_s3_timestamps=True,
            )

            mocked_get_credentials.assert_called_once()
            actual_args, actual_kwargs = mocked_client.call_args
            expected_args = tuple(["s3"])
            expected_kwargs = {
                "aws_access_key_id": credentials.access_key,
                "aws_secret_access_key": credentials.secret_key,
                "aws_session_token": credentials.token,
                "region_name": "us-west-2",
            }

            actual_total_args = actual_args + tuple(actual_kwargs.get(key, None) for key in expected_kwargs)
            expected_total_args = expected_args + tuple(expected_kwargs.values())

            assert len(actual_total_args) == len(expected_total_args)
            assert actual_total_args == expected_total_args
            assert actual_total_args == expected_total_args