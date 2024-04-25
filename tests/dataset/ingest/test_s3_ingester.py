"""Test suite for the `dataset.ingest.s3_ingester` module."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import moto
import pytest
from botocore.exceptions import ProfileNotFound
from moto.s3.models import s3_backends

from mleko.dataset.ingest import S3Ingester


class TestS3Ingester:
    """Test suite for `dataset.ingest.s3_ingester.S3Ingester`."""

    @pytest.fixture(scope="function")
    def s3_bucket(self):
        """Mock S3 Bucket."""
        with moto.mock_s3():
            s3 = boto3.resource("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            yield s3.Bucket("test-bucket")

    def test_download(self, s3_bucket, temporary_directory: Path):
        """Should download data to temp dir."""
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="test-file-data")

        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=False,
        )
        test_data.fetch_data(
            force_recompute=True,
        )

        assert (test_data._cache_directory / "test-file.csv").read_text() == "test-file-data"

    def test_different_timestamps(self, s3_bucket, temporary_directory: Path):
        """Should throw exception due to different timestamps of files."""
        s3_bucket.Object("test-prefix/manifest").put(Body='{"entries": []}')
        s3_bucket.Object("test-prefix/test-file.csv").put(Body="test-file-data")
        s3_backends["123456789012"]["global"].buckets["test-bucket"].keys[  # type: ignore
            "test-prefix/test-file.csv"
        ].last_modified = datetime.datetime(2000, 1, 1)

        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        with pytest.raises(Exception, match="Files in S3"):
            test_data.fetch_data(
                force_recompute=True,
            )

    def test_is_cached(self, s3_bucket, temporary_directory: Path):
        """Should use cached files if manifest matches with local files in temp dir."""
        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO")
        s3_bucket.Object("test-prefix/test-file2.csv").put(Body="MLEKO1")
        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        test_data.fetch_data(
            force_recompute=True,
        )

        with patch.object(S3Ingester, "_s3_fetch_all") as mocked_s3_fetch_all:
            test_data = S3Ingester(
                cache_directory=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                file_pattern="test-file1.csv",
                num_workers=1,
                check_s3_timestamps=False,
            )
            test_data.fetch_data(
                force_recompute=False,
            )
            mocked_s3_fetch_all.assert_not_called()

    def test_no_matching_files(self, s3_bucket, temporary_directory: Path):
        """Should raise `FileNotFoundError` if no files match the pattern."""
        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO1")
        s3_bucket.Object("test-prefix/test-file2.csv").put(Body="MLEKO2")
        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            file_pattern="test-file3.csv",
            num_workers=1,
            check_s3_timestamps=True,
        )

        with pytest.raises(FileNotFoundError):
            test_data.fetch_data()

    def test_is_outdated_cache(self, s3_bucket, temporary_directory: Path):
        """Should not use cached files if manifest does not match with local files in temp dir."""
        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO1")
        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        test_data.fetch_data(
            force_recompute=True,
        )

        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO_NEW")
        with patch.object(S3Ingester, "_s3_fetch_all") as mocked_s3_fetch_all, patch("os.path.getsize", return_value=3):
            test_data = S3Ingester(
                cache_directory=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                num_workers=1,
                check_s3_timestamps=False,
            )
            test_data.fetch_data(
                force_recompute=False,
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

            S3Ingester(
                cache_directory=temporary_directory,
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

    def test_missing_credentials(self, temporary_directory: Path):
        """Should init with custom aws_profile_name and aws_region_name."""
        with patch("boto3.Session.__init__") as mocked_session_init, patch(
            "boto3.Session.get_credentials"
        ) as mocked_get_credentials:
            # Return None for the default region_name and profile_name
            def side_effect(*args, **kwargs):
                if kwargs.get("region_name") == "us-west-2" and kwargs.get("profile_name") == "custom-profile-name":
                    kwargs["session"] = MagicMock()
                    return None
                raise ProfileNotFound(profile=kwargs.get("profile_name"))  # type: ignore

            mocked_session_init.side_effect = side_effect
            mocked_get_credentials.return_value = None

            with pytest.raises(ValueError):
                S3Ingester(
                    cache_directory=temporary_directory,
                    s3_bucket_name="test-bucket",
                    s3_key_prefix="test-prefix",
                    aws_profile_name="custom-profile-name",
                    aws_region_name="us-west-2",
                    num_workers=1,
                    check_s3_timestamps=True,
                )

            mocked_get_credentials.assert_called_once()

    def test_is_cached_nested(self, s3_bucket, temporary_directory: Path):
        """Should use cached files if manifest matches with local files in temp dir."""
        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO")
        s3_bucket.Object("test-prefix/test-file2.csv").put(Body="MLEKO1")
        s3_bucket.Object("test-prefix/nested/test-file2.csv").put(Body="MLEKO2")
        s3_bucket.Object("test-prefix/nested/test-file3.csv").put(Body="MLEKO3")
        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            num_workers=1,
            check_s3_timestamps=True,
        )
        test_data.fetch_data(
            force_recompute=True,
        )

        with patch.object(S3Ingester, "_s3_fetch_all") as mocked_s3_fetch_all:
            test_data = S3Ingester(
                cache_directory=temporary_directory,
                s3_bucket_name="test-bucket",
                s3_key_prefix="test-prefix",
                aws_region_name="us-east-1",
                file_pattern="*test-file2.csv",
                num_workers=1,
                check_s3_timestamps=False,
            )
            test_data.fetch_data(
                force_recompute=False,
            )
            mocked_s3_fetch_all.assert_not_called()

    def test_s3_manifest(self, s3_bucket, temporary_directory: Path):
        """Should use cached files if manifest matches with local files in temp dir."""
        s3_bucket.Object("test-prefix/test-file1.csv").put(Body="MLEKO")
        s3_bucket.Object("test-prefix/test-file2.csv").put(Body="MLEKO1")
        s3_bucket.Object("test-prefix/old-file.csv").put(Body="TEST")
        s3_bucket.Object("test-prefix/manifest").put(
            Body=json.dumps(
                {
                    "entries": [
                        {
                            "url": "s3://test-bucket/test-prefix/test-file1.csv",
                            "meta": {"content_length": 5},
                        },
                        {
                            "url": "s3://test-bucket/test-prefix/test-file2.csv",
                            "meta": {"content_length": 6},
                        },
                    ]
                }
            )
        )
        test_data = S3Ingester(
            cache_directory=temporary_directory,
            s3_bucket_name="test-bucket",
            s3_key_prefix="test-prefix",
            aws_region_name="us-east-1",
            manifest_file_name="manifest",
            dataset_id="test-dataset",
            num_workers=1,
            check_s3_timestamps=True,
        )
        file_paths = test_data.fetch_data(
            force_recompute=True,
        )
        assert len(file_paths) == 2
        assert (temporary_directory / "test-dataset" / "test-file1.csv").read_text() == "MLEKO"
        assert (temporary_directory / "test-dataset" / "test-file2.csv").read_text() == "MLEKO1"
