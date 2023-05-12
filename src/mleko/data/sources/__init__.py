"""Provides functionality for fetching data from various sources such as AWS S3 and Kaggle, and storing it locally.

This subpackage contains classes designed to easily fetch data from different sources, like AWS S3 or Kaggle,
and store them locally in specified destination directories. The main classes are 'BaseDataSource', an abstract base
class for implementing specific data source classes, along with concrete implementations.
"""
from __future__ import annotations

from .base_data_source import BaseDataSource
from .kaggle_data_source import KaggleDataSource
from .s3_data_source import S3DataSource


__all__ = ["BaseDataSource", "S3DataSource", "KaggleDataSource"]
