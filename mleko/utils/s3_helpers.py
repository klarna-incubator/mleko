"""This module contains helper functions for working with AWS S3."""

from __future__ import annotations

import boto3
from botocore.config import Config as BotoConfig

from .custom_logger import CustomLogger


logger = CustomLogger()
"""A module-level custom logger."""


def get_s3_client(
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
