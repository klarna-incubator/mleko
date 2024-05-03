"""The module contains a fingerprinter for JSON data."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .base_fingerprinter import BaseFingerprinter


class JsonFingerprinter(BaseFingerprinter):
    """Class to generate unique fingerprints for valid JSON data."""

    def fingerprint(self, data: dict[str, Any] | list[Any] | None) -> str:
        """Generate a fingerprint string for a given JSON.

        Args:
            data: The JSON data to generate a fingerprint for.

        Returns:
            A fingerprint that uniquely identifies the JSON data.
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
