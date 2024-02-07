"""The subpackage provides functionality for fetching data from various sources.

This subpackage contains classes designed to easily fetch data from different sources, like AWS S3 or Kaggle,
and store them locally in specified destination directories. The main classes are 'BaseIngester', an abstract base
class for implementing specific data source classes, along with concrete implementations.

The following ingester classes are provided by the subpackage:
    - `BaseIngester`: The abstract base class for all ingesters.
        - `LocalFileEntry`: A class representing a local file entry.
        - `LocalManifest`: A class representing a local manifest.
        - `LocalManifestHandler`: A class for handling local manifests.
    - `S3Ingester`: An ingester for fetching data from AWS S3.
    - `KaggleIngester`: An ingester for fetching data from Kaggle.
"""

from __future__ import annotations

from .base_ingester import BaseIngester, LocalFileEntry, LocalManifest, LocalManifestHandler
from .kaggle_ingester import KaggleIngester
from .s3_ingester import S3Ingester


__all__ = ["BaseIngester", "S3Ingester", "KaggleIngester", "LocalFileEntry", "LocalManifest", "LocalManifestHandler"]
