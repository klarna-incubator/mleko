"""Test suite for the `dataset.export.base_exporter` module."""

from __future__ import annotations

from pathlib import Path

from mleko.dataset.export.base_exporter import BaseExporter


class TestBaseExporter:
    """Test suite for `dataset.export.base_exporter.BaseExporter`."""

    class StringExporter(BaseExporter):
        """Test class inheriting from the `BaseExporter`."""

        def export(self, data, file_path, force_recompute):
            """Export data."""
            with open(file_path, "w") as file:
                file.write(data)

            return file_path

    def test_export(self, temporary_directory: Path):
        """Should export the data to a file."""
        test_exporter = self.StringExporter()
        file_path = test_exporter.export("test data", temporary_directory / "test.txt", False)
        assert file_path.exists()
        assert file_path.read_text() == "test data"
