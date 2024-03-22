"""This package provides Fingerprinter classes for generating unique fingerprints.

Fingerprinters are used for generating unique fingerprints of various data and file types,
such as Vaex DataFrames or CSV files. These fingerprints can be used to track changes in data and support
caching mechanisms.

The following fingerprinting utilities are provided:
    - `BaseFingerprinter`: The base class for all fingerprinters.
    - `CSVFingerprinter`: A fingerprinter for CSV files.
    - `VaexFingerprinter`: A fingerprinter for Vaex DataFrames.
    - `DictFingerprinter`: A fingerprinter for dictionaries.
    - `CallableSourceFingerprinter`: A fingerprinter for Python Callables that hashes the source code of the Callable
        to generate a fingerprint.
    - `OptunaSamplerFingerprinter`: A fingerprinter for Optuna samplers.
    - `OptunaPrunerFingerprinter`: A fingerprinter for Optuna pruners.
"""

from __future__ import annotations

from .base_fingerprinter import BaseFingerprinter
from .callable_source_fingerprinter import CallableSourceFingerprinter
from .csv_fingerprinter import CSVFingerprinter
from .dict_fingerprinter import DictFingerprinter
from .optuna_pruner_fingerprinter import OptunaPrunerFingerprinter
from .optuna_sampler_fingerprinter import OptunaSamplerFingerprinter
from .vaex_fingerprinter import VaexFingerprinter


__all__ = [
    "BaseFingerprinter",
    "CallableSourceFingerprinter",
    "CSVFingerprinter",
    "VaexFingerprinter",
    "OptunaPrunerFingerprinter",
    "OptunaSamplerFingerprinter",
    "DictFingerprinter",
]
