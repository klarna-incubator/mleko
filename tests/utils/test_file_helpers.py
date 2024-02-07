"""Test suite for the `utils.file_helpers` module."""

from __future__ import annotations

from pathlib import Path

from mleko.utils.file_helpers import clear_directory


class TestClearDirectory:
    """Test suite for `utils.file_helpers.clear_directory`."""

    def test_clear_directory(self, temporary_directory: Path):
        """Should delete all files."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()

        clear_directory(temporary_directory)
        remaining_files = list(temporary_directory.iterdir())
        assert len(remaining_files) == 0

    def test_with_pattern(self, temporary_directory: Path):
        """Should only delete pattern matched files."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()
            (temporary_directory / f"file{i}.log").touch()

        clear_directory(temporary_directory, "*.log")
        remaining_files_log = list(temporary_directory.glob("*.log"))
        assert len(remaining_files_log) == 0

        remaining_files_txt = list(temporary_directory.glob("*.txt"))
        assert len(remaining_files_txt) == 3

    def test_no_files(self, temporary_directory: Path):
        """Should handle empty directory."""
        assert len(list(temporary_directory.iterdir())) == 0

        clear_directory(temporary_directory)

    def test_no_matching_pattern(self, temporary_directory: Path):
        """Should handle no files matching pattern."""
        for i in range(3):
            (temporary_directory / f"file{i}.txt").touch()

        assert len(list(temporary_directory.glob("*.txt"))) == 3

        clear_directory(temporary_directory, "*.log")
