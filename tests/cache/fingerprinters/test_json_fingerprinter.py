"""Test suite for the `cache.fingerprinters.json_fingerprinter`."""

from __future__ import annotations

from mleko.cache.fingerprinters.json_fingerprinter import JsonFingerprinter


class TestJsonFingerprinter:
    """Test suite for `cache.fingerprinters.json_fingerprinter.JsonFingerprinter`."""

    def test_empty_dict(self):
        """Should produce the same output over multiple runs."""
        json_obj = {}

        original_fingerprint = JsonFingerprinter().fingerprint(json_obj)
        new_fingerprint = JsonFingerprinter().fingerprint(json_obj)

        assert original_fingerprint == new_fingerprint

    def test_flat_json(self):
        """Should produce the same output over multiple runs."""
        json_obj = {"a": 1, "b": 2}

        original_fingerprint = JsonFingerprinter().fingerprint(json_obj)
        new_fingerprint = JsonFingerprinter().fingerprint(json_obj)

        assert original_fingerprint == new_fingerprint

    def test_flat_json_unsorted(self):
        """Should produce the same output over multiple runs."""
        json_obj_1 = {"b": 2, "a": 1}
        json_obj_2 = {"a": 1, "b": 2}

        fingerprint_1 = JsonFingerprinter().fingerprint(json_obj_1)
        fingerprint_2 = JsonFingerprinter().fingerprint(json_obj_2)

        assert fingerprint_1 == fingerprint_2

    def test_empty_list(self):
        """Should produce the same output over multiple runs."""
        json_obj = []

        original_fingerprint = JsonFingerprinter().fingerprint(json_obj)
        new_fingerprint = JsonFingerprinter().fingerprint(json_obj)

        assert original_fingerprint == new_fingerprint

    def test_filled_list(self):
        """Should produce the same output over multiple runs."""
        json_obj = [1, 2, 3]

        original_fingerprint = JsonFingerprinter().fingerprint(json_obj)
        new_fingerprint = JsonFingerprinter().fingerprint(json_obj)

        assert original_fingerprint == new_fingerprint

    def test_list_unsorted(self):
        """Should produce the same output over multiple runs."""
        json_obj_1 = [3, 1, 2]
        json_obj_2 = [1, 2, 3]

        fingerprint_1 = JsonFingerprinter().fingerprint(json_obj_1)
        fingerprint_2 = JsonFingerprinter().fingerprint(json_obj_2)

        assert fingerprint_1 != fingerprint_2

    def test_nested_json_unsorted(self):
        """Should produce the same output over multiple runs."""
        json_obj_1 = {"b": 2, "a": {"c": [1, 2, 3], "b": 2}}
        json_obj_2 = {"a": {"b": 2, "c": [1, 2, 3]}, "b": 2}

        fingerprint_1 = JsonFingerprinter().fingerprint(json_obj_1)
        fingerprint_2 = JsonFingerprinter().fingerprint(json_obj_2)

        assert fingerprint_1 == fingerprint_2

    def test_nested_json_unsorted_list_difference(self):
        """Should produce the same output over multiple runs."""
        json_obj_1 = {"b": 2, "a": {"c": [1, 3, 2], "b": 2}}
        json_obj_2 = {"a": {"b": 2, "c": [1, 2, 3]}, "b": 2}

        fingerprint_1 = JsonFingerprinter().fingerprint(json_obj_1)
        fingerprint_2 = JsonFingerprinter().fingerprint(json_obj_2)

        assert fingerprint_1 != fingerprint_2

    def test_list_of_dicts_unsorted(self):
        """Should produce the same output over multiple runs."""
        json_obj_1 = [{"b": 2, "a": 1}, {"b": 3, "a": 2}]
        json_obj_2 = [{"a": 1, "b": 2}, {"a": 2, "b": 3}]

        fingerprint_1 = JsonFingerprinter().fingerprint(json_obj_1)
        fingerprint_2 = JsonFingerprinter().fingerprint(json_obj_2)

        assert fingerprint_1 == fingerprint_2
