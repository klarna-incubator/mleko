"""Test suite for the `cache.fingerprinters.callable_source_fingerprinter`."""
from __future__ import annotations

from mleko.cache.fingerprinters.callable_source_fingerprinter import CallableSourceFingerprinter


class TestCallableSourceFingerprinter:
    """Test suite for `cache.fingerprinters.callable_source_fingerprinter.CallableSourceFingerprinter`."""

    def test_stable_output(self):
        """Should produce the same output over multiple runs."""

        def test_func():
            return 1

        original_fingerprint = CallableSourceFingerprinter().fingerprint(test_func)
        new_fingerprint = CallableSourceFingerprinter().fingerprint(test_func)

        assert original_fingerprint == new_fingerprint

    def test_detects_single_change(self):
        """Should detect a change in the source code of the function."""

        def test_func():  # type: ignore
            return 1

        original_fingerprint = CallableSourceFingerprinter().fingerprint(test_func)

        def test_func():
            return 2

        new_fingerprint = CallableSourceFingerprinter().fingerprint(test_func)

        assert original_fingerprint != new_fingerprint
