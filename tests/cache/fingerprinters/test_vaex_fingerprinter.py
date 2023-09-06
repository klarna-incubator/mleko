"""Test suite for `cache.fingerprinters.vaex_fingerprinter`."""
from __future__ import annotations

import pytest
import vaex

from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter


class TestVaexFingerprinter:
    """Test suite for `cache.fingerprinters.vaex_fingerprinter.VaexFingerprinter`."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate csv files for testing."""
        self.data = {"column1": [1, 2, 3], "column2": ["A", "B", "C"], "column3": [0.1, 0.2, 0.3]}
        self.vaex_fingerprinter = VaexFingerprinter()

    def test_stable_output(self):
        """Should produce the same output over multiple runs even when."""
        df = vaex.from_dict(self.data)
        original_fingerprint = self.vaex_fingerprinter.fingerprint(df)
        new_fingerprint = self.vaex_fingerprinter.fingerprint(df)

        assert original_fingerprint == new_fingerprint

    def test_on_different_n_rows(self):
        """Should produce same fingerprint on two dataframes with same content."""
        original_df = vaex.from_dict(self.data)
        original_fingerprint = self.vaex_fingerprinter.fingerprint(original_df)

        new_df = vaex.from_dict(self.data)
        new_fingerprint = self.vaex_fingerprinter.fingerprint(new_df)

        assert original_fingerprint == new_fingerprint

    def test_detects_single_change(self):
        """Should have different fingerprint on small change to DataFrame."""
        original_df = vaex.from_dict(self.data)
        original_fingerprint = self.vaex_fingerprinter.fingerprint(original_df)

        self.data["column1"][0] = 2
        new_df = vaex.from_dict(self.data)
        new_fingerprint = self.vaex_fingerprinter.fingerprint(new_df)

        assert original_fingerprint != new_fingerprint
