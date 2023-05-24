"""Provides functionality for fetching data from various sources such as AWS S3 and Kaggle, and storing it locally.

This subpackage contains classes designed to easily fetch data from different sources, like AWS S3 or Kaggle,
and store them locally in specified destination directories. The main classes are 'BaseIngester', an abstract base
class for implementing specific data source classes, along with concrete implementations.
"""
from __future__ import annotations

from .base_ingester import BaseIngester
from .kaggle_ingester import KaggleIngester
from .s3_ingester import S3Ingester


__all__ = ["BaseIngester", "S3Ingester", "KaggleIngester"]
