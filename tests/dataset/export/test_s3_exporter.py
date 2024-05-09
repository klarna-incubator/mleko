"""Test suite for the `dataset.export.s3_exporter` module."""

import json
from pathlib import Path
from unittest.mock import patch

import boto3
import moto
import pytest

from mleko.dataset.export import S3Exporter


class TestS3Exporter:
    """Test suite for `dataset.export.s3_exporter.S3Exporter`."""

    @pytest.fixture(scope="function")
    def s3_bucket(self):
        """Mock S3 Bucket."""
        with moto.mock_s3():
            s3 = boto3.resource("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            yield s3.Bucket("test-bucket")

    def test_upload(self, s3_bucket, temporary_directory: Path):
        """Should upload data to S3."""
        file_path = temporary_directory / "test-file.csv"
        file_path.write_text("test-file-data")

        s3_exporter = S3Exporter(
            aws_region_name="us-east-1",
            max_concurrent_files=1,
        )
        s3_paths = s3_exporter.export(
            [file_path], {"bucket_name": "test-bucket", "key_prefix": "data/", "extra_args": None}
        )

        assert s3_paths == [f"s3://test-bucket/data/{file_path.name}"]
        assert s3_bucket.Object(f"data/{file_path.name}").get()["Body"].read().decode() == "test-file-data"

    def test_cached_upload_ensure_manifest_creation(self, s3_bucket, temporary_directory: Path):
        """Should skip upload if files are already in S3."""
        file_path = temporary_directory / "test-file.csv"
        file_path.write_text("test-file-data")
        s3_bucket.put_object(Key="data/test-file.csv", Body="test-file-data")

        s3_exporter = S3Exporter(
            aws_region_name="us-east-1",
            max_concurrent_files=1,
        )

        with patch.object(S3Exporter, "_s3_export_all") as mock_s3_export_all:
            s3_paths = s3_exporter.export(
                [file_path], {"bucket_name": "test-bucket", "key_prefix": "data/", "extra_args": None}
            )

            assert s3_paths == [f"s3://test-bucket/data/{file_path.name}"]
            assert json.loads(s3_bucket.Object("data/manifest").get()["Body"].read().decode()) == {
                "entries": [{"url": f"data/{file_path.name}", "meta": {"content_length": 14}}]
            }

            mock_s3_export_all.assert_not_called()

    def test_cached_subset_upload_and_manifest_update(self, s3_bucket, temporary_directory: Path):
        """Should skip upload if files are already in S3."""
        file_path_1 = temporary_directory / "test-file1.csv"
        file_path_1.write_text("test-file-data")
        file_path_2 = temporary_directory / "test-file2.csv"
        file_path_2.write_text("test-file-data")
        s3_bucket.put_object(Key="data/test-file1.csv", Body="test-file-data")
        s3_bucket.put_object(Key="data/test-file2.csv", Body="test-file-data")
        s3_bucket.put_object(
            Key="data/manifest",
            Body=json.dumps(
                {
                    "entries": [
                        {"url": "data/test-file1.csv", "meta": {"content_length": 14}},
                        {"url": "data/test-file2.csv", "meta": {"content_length": 14}},
                    ]
                }
            ),
        )

        s3_exporter = S3Exporter(
            aws_region_name="us-east-1",
            max_concurrent_files=1,
        )

        with patch.object(S3Exporter, "_s3_export_all") as mock_s3_export_all:
            s3_paths = s3_exporter.export(
                [file_path_1], {"bucket_name": "test-bucket", "key_prefix": "data/", "extra_args": None}
            )

            assert s3_paths == [f"s3://test-bucket/data/{file_path_1.name}"]
            assert json.loads(s3_bucket.Object("data/manifest").get()["Body"].read().decode()) == {
                "entries": [{"url": f"data/{file_path_1.name}", "meta": {"content_length": 14}}]
            }

            mock_s3_export_all.assert_not_called()

    def test_cached_forced_recompute(self, s3_bucket, temporary_directory: Path):
        """Should upload data to S3."""
        file_path = temporary_directory / "test-file.csv"
        file_path.write_text("test-file-date")
        s3_bucket.put_object(Key="data/test-file.csv", Body="test-file-data")

        s3_exporter = S3Exporter(
            aws_region_name="us-east-1",
            max_concurrent_files=1,
        )
        s3_paths = s3_exporter.export(
            [file_path], {"bucket_name": "test-bucket", "key_prefix": "data/", "extra_args": None}, force_recompute=True
        )

        assert s3_paths == [f"s3://test-bucket/data/{file_path.name}"]
        assert s3_bucket.Object(f"data/{file_path.name}").get()["Body"].read().decode() == "test-file-date"
