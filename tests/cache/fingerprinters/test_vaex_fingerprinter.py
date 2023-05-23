"""Test suite for `cache.fingerprinters.vaex_fingerprinter`."""
from __future__ import annotations

from pathlib import Path

import vaex
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter


class TestVaexFingerprinter:
    """Test suite for `cache.fingerprinters.vaex_fingerprinter.VaexFingerprinter`."""

    def test_stable_output(self):
        """Should produce the same output over multiple runs even when."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(df)
        new_fingerprint = VaexFingerprinter().fingerprint(df)
        assert original_fingerprint == new_fingerprint

    def test_on_different_n_rows(self, temporary_directory: Path):
        """Should produce same fingerprint on two dataframes with same content."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        original_df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(original_df)

        new_df = vaex.from_dict(data)
        new_fingerprint = VaexFingerprinter().fingerprint(new_df)
        assert original_fingerprint == new_fingerprint

    def test_detects_single_change(self):
        """Should have different fingerprint on small change to DataFrame."""
        data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        original_df = vaex.from_dict(data)
        original_fingerprint = VaexFingerprinter().fingerprint(original_df)

        data["column1"][0] = 2
        new_df = vaex.from_dict(data)
        new_fingerprint = VaexFingerprinter().fingerprint(new_df)
        assert original_fingerprint != new_fingerprint
