"""MLEKO (ML Ekosystem): Your Complete Solution for Streamlined Model Building.

MLEKO is a comprehensive Python library covering the entire model building process, from data fetching
and cleaning to generating model performance reports. It provides an end-to-end solution to streamline the model
building process while ensuring efficient use of resources and time.

The library is organized into various subpackages, each focusing on a specific aspect of the model building process:

* Pipeline: A subpackage dedicated to managing and executing customizable data processing pipelines with pre-built
   pipeline steps for various data processing tasks.

* Data Processing: A subpackage that handles data fetching, conversion, filtering, and feature engineering for
   different data sources and formats.

* Caching and Fingerprinting: A subpackage providing caching mixins and fingerprinting utilities for efficient
   caching of method call results and tracking changes in data.

* Utilities: A collection of utility functions for logging, decorating, file management, and TQDM wrappers.

Each subpackage is designed to be modular and extensible, making it easy to customize and adapt the library to a wide
range of model building processes and requirements.
"""
from __future__ import annotations


__version__ = "0.7.0"
