"""The module contains a fingerprinter for dictionaries."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mleko.utils.custom_logger import CustomLogger

from .base_fingerprinter import BaseFingerprinter


logger = CustomLogger()
"""The logger for the module."""


class DictFingerprinter(BaseFingerprinter):
    """Class to generate unique fingerprints for dictionaries."""

    def fingerprint(self, data: dict[str, Any] | None) -> str:
        """Generate a fingerprint string for a given dictionary.

        Args:
            data: The dictionary to dump.

        Returns:
            A fingerprint that uniquely identifies the dictionary.
        """

        def deep_sort(obj: dict | Any):
            """Recursively sort nested dicts.

            Args:
                obj: The dict or object to sort.

            Returns:
                Sorted dict, or object.
            """
            if isinstance(obj, dict):
                _sorted = {}
                for key in sorted(obj):
                    _sorted[key] = deep_sort(obj[key])
                return _sorted
            else:
                return obj

        sorted_data = deep_sort(data)
        return hashlib.md5((json.dumps(sorted_data, sort_keys=True) if data is not None else "").encode()).hexdigest()
