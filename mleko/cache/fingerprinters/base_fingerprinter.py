"""This module contains the abstract base class for creating specialized fingerprinters.

The fingerprinter is used to generate a unique identifier for the given data, which is used
to detect changes in the data. The fingerprinter is used by the cache to determine whether
the data has changed since the last time it was cached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseFingerprinter(ABC):
    """Abstract base class for creating specialized fingerprinters."""

    @abstractmethod
    def fingerprint(self, data: Any) -> str:
        """Generate a fingerprint for the given data.

        The fingerprint should be a unique identifier for the given data, across different
        runs of the program, i.e. the fingerprint should be the same for the same data
        regardless of when the program is run.

        Args:
            data: Data that should be fingerprinted.

        Raises:
            NotImplementedError: The method has to be implemented by the subclass.

        Returns:
            str: The fingerprint as a hexadecimal string.
        """
        raise NotImplementedError
